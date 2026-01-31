import os
import re

from utils import *

class TokenDataLoader:
    def __init__(self, filename = os.path.join(file_path, TINY_SHAKESPEARE_FILENAME)):

        if not os.path.exists(filename):
            log_erro(f'dataset file not exists, please download it first: {filename}')
            raise Exception('file for training not exists')
        
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

        tokens = [t for t in text.split(' ') if t != '']
        
        freqs = {}
        for token in tokens:
            if token in freqs:
                freqs[token] += 1
            else:
                freqs[token] = 1
        
        tokens = [t if freqs[t] > 1 else '<UNK>' for t in tokens]
        unified_tokens = sorted(list(set(tokens)))
        self.word_cnt = len(unified_tokens)  # how many deduplicate words
        self.token_cnt = len(tokens)         # how many tokens
        
        self.word_to_ix = {token: ix for ix, token in enumerate(unified_tokens)}
        self.ix_to_word = {ix: token for ix, token in enumerate(unified_tokens)}
        
        self.token_ixs = [self.word_to_ix[token] for token in tokens]

        # for providing
        self.seq_len = 64 # providing 64 tokens in a time
    
    def get_tokens(self):
        
        pointer = 0
        while pointer + self.seq_len + 1 <= self.token_cnt:
            # input and output but offset 1
            seq_token = self.token_ixs[pointer: pointer+self.seq_len+1]

            yield seq_token, False

            pointer += self.seq_len
        
        yield None, True

        
        