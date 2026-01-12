"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    letter_x = sum(row.count(X) for row in board)
    letter_o = sum(row.count(O) for row in board)
    if letter_x > letter_o:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    moves = set()
    
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == EMPTY:
                moves.add((i, j))
    
    return moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    new_board = copy.deepcopy(board)
    letter = player(board)
    new_board[action[0]][action[1]] = letter
    
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for i in range(len(board)):
        if board[i][0] != EMPTY and board[i][0] == board[i][1] == board[i][2]:
            return board[i][0]
        elif board[0][i] != EMPTY and board[0][i] == board[1][i] == board[2][i]:
            return board[0][i]
        
    if board[1][1] != EMPTY and (board[0][0] == board[1][1] == board[2][2] or board[0][2] == board[1][1] == board[2][0]):
        return board[1][1]
    
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    empty_squares = sum(row.count(EMPTY) for row in board)
    
    for i in range(len(board)):
        if (board[i][0] != EMPTY and board[i][0] == board[i][1] == board[i][2]) or (board[0][i] != EMPTY and board[0][i] == board[1][i] == board[2][i]):
            return True
    if board[1][1] != EMPTY and (board[0][0] == board[1][1] == board[2][2] or board[0][2] == board[1][1] == board[2][0]):
        return True   
    elif not empty_squares:
        return True
    
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    letter =  winner(board)
    
    if letter == X:
        return 1
    elif letter == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    elif sum(row.count(EMPTY) for row in board) == 9:
        return (1, 1)
    
    def max_value(board):
        if terminal(board):
            return utility(board), None
        v = -1000
        best_move = None
        for action in actions(board):
            min_result, _ = min_value(result(board, action))
            if min_result > v:
                v = min_result
                best_move = action
        return v, best_move
    
    def min_value(board):
        if terminal(board):
            return utility(board), None
        v = 1000
        best_move = None
        for action in actions(board):
            max_result, _ = max_value(result(board, action))
            if max_result < v:
                v = max_result
                best_move = action
        return v, best_move
    
    if player(board) == X:
        _, move = max_value(board)
    else:
        _, move = min_value(board)
        
    return move
