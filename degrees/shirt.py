def main():
    arr = [87,101,39,114,101,32,112,114,101,103,110,97,110,116,33]
    msg = [chr(i) for i in arr]
    print(''.join(msg))
    
if __name__ == '__main__':
    main()