
import os

from constants import *

def main():
    filename = os.path.join(file_path, TINY_SHAKESPEARE_FILENAME)
    with open(filename, 'r') as pf:
        text = pf.read()
        vocab = sorted(list(set(text)))
        print(vocab)
        print(len(vocab))
        print(len(text))

if __name__ == '__main__':
    main()
