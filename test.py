import os
import re

from constants import *

def main1():
    filename = os.path.join(file_path, TINY_SHAKESPEARE_FILENAME)
    with open(filename, 'r') as pf:
        text = pf.read()
        vocab = sorted(list(set(text)))
        print(vocab)
        print(len(vocab))
        print(len(text))

def main2():
    filename = os.path.join(file_path, TINY_SHAKESPEARE_FILENAME)
    with open(filename, 'r') as pf:
        text = pf.read()
    
    text = re.sub(r"n't", " n't", text)
    text = re.sub(r"'s", " 's", text)
    text = re.sub(r"'ve", " 've", text)
    text = re.sub(r"'re", " 're", text)
    text = re.sub(r"'ll", " 'll", text)
    text = re.sub(r"'d", " 'd", text)
    text = re.sub(r"'m", " 'm", text)
    
    pattern = "([.,!?;:\n])"
    text = re.sub(pattern, r" \1 ", text)

    tokenized_filename = os.path.join(file_path, TOKENIZED_TINY_SHAKESPEARE_FILENAME)
    with open(tokenized_filename, 'w') as pf:
        pf.write(text)
    
def main3():
    tokenized_filename = os.path.join(file_path, TOKENIZED_TINY_SHAKESPEARE_FILENAME)
    with open(tokenized_filename, 'r') as pf:
        text = pf.read()
        tokens = [t for t in text.split(' ') if t != '']
        # print(tokens[:500])
        freqs = {}
        for token in tokens:
            if token in freqs:
                freqs[token] += 1
            else:
                freqs[token] = 1
        tokens = [t if freqs[t] > 1 else '<UNK>' for t in tokens]
        unified_tokens = sorted(list(set(tokens)))
        word_cnt = len(unified_tokens)
        
        word_to_ix = {}
        ix_to_word = [''] * word_cnt

        for ix, token in enumerate(unified_tokens):
            word_to_ix[token] = ix
            ix_to_word[ix] = token
        
        token_ixs = []
        for token in tokens:
            token_ixs.append(word_to_ix[token])


if __name__ == '__main__':
    # main1()
    main2()
    main3()

