
import os

from utils import *

class DataLoader:
    def __init__(self, filename = os.path.join(file_path, TINY_SHAKESPEARE_FILENAME)):

        if not os.path.exists(filename):
            log_erro(f'dataset file not exists, please download it first: {filename}')
            raise Exception('file for training not exists')
        
        with open(filename, 'r') as pf:
            self.text = pf.read()
        
        chars = sorted(list(set(self.text)))
        self.char_to_ix = {ch:i for i, ch in enumerate(chars)}
        self.ix_to_char = {i:ch for i, ch in enumerate(chars)}
        self.dic_len = len(chars)

        self.ixs = np.array([self.char_to_ix[ch] for ch in self.text], dtype=np.int32)
        
        self.text_len = len(self.text)
        self.sentence_len = sentence_len

        self.sentence_batch_size = sentence_batch_size

        log_info(f'file {filename} is readed. ready for providing sentences for training')

    def get_sentences(self):
        pointer = 0
        while True:
            if pointer + self.sentence_len + 1 > self.text_len:
                pointer = 0
                yield None, None, True 
                continue

            inputs = self.ixs[pointer : pointer + self.sentence_len]
            targets = self.ixs[pointer + 1 : pointer + self.sentence_len + 1]
            
            yield inputs, targets, False
            
            pointer += self.sentence_len
    
    # inputs  (bs, 100)
    # targets (bs, 100)
    # is_epoch_end
    def get_sentences_batch(self):
            n_step = (self.text_len - 1) // (self.sentence_batch_size * self.sentence_len)
            rounded_len = n_step * self.sentence_batch_size * self.sentence_len
            
            data_x = self.ixs[:rounded_len].reshape(self.sentence_batch_size, -1)
            data_y = self.ixs[1:rounded_len + 1].reshape(self.sentence_batch_size, -1)

            pointer = 0
            max_pointer = data_x.shape[1]

            while True:
                if pointer + self.sentence_len > max_pointer:
                    pointer = 0
                    yield None, None, True
                    continue
            
                inputs = data_x[:, pointer : pointer + self.sentence_len]
                targets = data_y[:, pointer : pointer + self.sentence_len]

                yield inputs, targets, False

                pointer += self.sentence_len

