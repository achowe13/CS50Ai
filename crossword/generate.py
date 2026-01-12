import sys
from collections import deque

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        print(letters)
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):  
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for variable in self.crossword.variables:
            for word in self.crossword.words:
                if len(word) != variable.length:
                    self.domains[variable].remove(word)

    def revise(self, x, y):  
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        
        joint_cell = [cell for cell in x.cells if cell in y.cells] 
        if joint_cell:
            x_index = x.cells.index(joint_cell[0])
            y_index = y.cells.index(joint_cell[0])
        
        dom_x_copy = self.domains[x].copy()
        for word in dom_x_copy:
            if len(self.domains[y]) == 1 and word in self.domains[y]:
                self.domains[x].remove(word)
                revised = True
            elif joint_cell:
                x_index = x.cells.index(joint_cell[0])
                y_index = y.cells.index(joint_cell[0])
                match = False
                for y_word in self.domains[y]:
                    if word[x_index] == y_word[y_index]:
                        match = True
                        break
                if not match:
                    self.domains[x].remove(word)
                    revised = True
        
        return revised

    def ac3(self, arcs=None):  
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        queue = None
        if arcs:
            queue = deque(arcs)
        else:
            queue = deque([])
            for x in self.crossword.variables:
                for y in self.crossword.neighbors(x):
                    queue.append((x, y))
            
        while queue:
            x, y = queue.popleft()
            if self.revise(x, y):
                if not self.domains[x]:
                    return False
                neighbors = self.crossword.neighbors(x)
                for z in neighbors - {y}:
                    queue.append((z, x))
        return True

    def assignment_complete(self, assignment):   
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        if len(assignment) == len(self.domains):
            return True
        return False

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        words = set()
        if not assignment:
            return True
        for key, value in assignment.items():
            if len(value) == key.length and value not in words:
                words.add(value)
            else:
                return False
            for neighbor in self.crossword.neighbors(key):
                if neighbor in assignment:
                    overlap = self.crossword.overlaps[key, neighbor]
                    if overlap:
                        i, j = overlap
                        if value[i] != assignment[neighbor][j]:
                            return False
                    
        return True

    def order_domain_values(self, var, assignment): 
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        least_constraining_values = []
        neighbors = self.crossword.neighbors(var)
        for var_word in self.domains[var]:
            num_neighbors_inpacted = 0
            for neighbor in neighbors:
                if neighbor in assignment:
                    continue
                overlap = self.crossword.overlaps[var, neighbor]
                if overlap:
                    i, j = overlap
                for word in self.domains[neighbor]:
                    if var_word[i] != word[j]:
                        num_neighbors_inpacted += 1
                        
            least_constraining_values.append((var_word, num_neighbors_inpacted))
        
        sorted_LCVs = sorted(least_constraining_values, key=lambda impacted: impacted[1])
        sorted_variable_list = [variable for variable, _ in sorted_LCVs]
        
        return sorted_variable_list

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassigned_vars = [var for var in self.domains if var not in assignment]

        min_var = None
        
        for var in unassigned_vars:
            if not min_var:
                min_var = unassigned_vars[0]
            elif len(self.domains[var]) < len(self.domains[min_var]):
                min_var = var
            elif len(self.domains[var]) == len(self.domains[min_var]):
                min_var_neighbors = self.crossword.neighbors(var)
                var_neighbors = self.crossword.neighbors(var)                
                if len(var_neighbors) > len(min_var_neighbors):
                    min_var = var
        
        return min_var

    def backtrack(self, assignment):  
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """        
        if self.assignment_complete(assignment):
            return assignment
        
        variable = self.select_unassigned_variable(assignment)
        domain_values = self.order_domain_values(variable, assignment)
        
        for value in domain_values:
            assignment[variable] = value
            if self.consistent(assignment):
                result = self.backtrack(assignment)
                if result is not None:
                    return result
            del assignment[variable]
        
        return None

        


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
