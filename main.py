import numpy as np
from typing import List, Dict, Tuple
import os

from constants import *
from dataloader import DataLoader
from utils import *

# vanilla RNN + adagrad
# always forgets memories
class MyShakespeare:
    def __init__(self, loader: DataLoader):
        
        #
        self.loader = loader
        self.dic_len = self.loader.dic_len # 65 as value
        
        self.epochs = epochs

        # hyperparameter
        self.n = n
        self.learning_rate = learning_rate

        # for sampling
        self.prompts = prompts

        # hidden parameters
        # the parameter between cells
        self.whh = self._orthogonal_initialize(self.n, self.n)
        self.dwhh = None
        # the parameter of input
        self.wxh = self._xavier_initialize(self.n, self.dic_len)
        self.dwxh = None
        # the parameter of output
        self.wyh = self._xavier_initialize(self.dic_len, self.n)
        self.dwyh = None

        # bias
        self.bh = self._zero_initialize(self.n, 1)
        self.dbh = None
        self.by = self._zero_initialize(self.dic_len, 1)
        self.dby = None

        # for adagrad
        self.mwhh = np.zeros_like(self.whh)
        self.mwxh = np.zeros_like(self.wxh)
        self.mwyh = np.zeros_like(self.wyh)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)

        # store some state from forward
        self.xs = None
        self.ps = None
        self.hs = None

        # store state of last sentence
        self.h_neg1 = self._zero_initialize(self.n, 1)

        # total parameter count
        tot_params = self.wxh.size + self.whh.size + self.wyh.size + self.bh.size + self.by.size
        log_info(f'the RNN model has a total of {tot_params} parameters1')

    def _orthogonal_initialize(self, *shape):
        return np.random.randn(*shape) * 0.01

    # two parameters required
    def _xavier_initialize(self, *shape):
        return np.random.randn(*shape) / np.sqrt(shape[1])

    def _zero_initialize(self, *shape):
        return np.zeros(shape)
    
    # generate one hot
    def one_hot(self, ix: int) -> np.array:
        oh = np.zeros((self.dic_len, 1))
        oh[ix] = 1
        return oh

    #                                                 xs    hs    ps    loss
    def _forward(self, sentence: List[int]) -> Tuple[Dict, Dict, Dict, float]:
        self.xs, self.hs, self.ps = {}, {}, {}
        loss = 0

        self.hs[-1] = self.h_neg1.copy()
        for t in range(len(sentence)):
            self.xs[t] = self.one_hot(sentence[t])
            
            # the state of next cell
            self.hs[t] = np.tanh(self.wxh @ self.xs[t] + self.whh @ self.hs[t-1] + self.bh)
            y = self.wyh @ self.hs[t] + self.by
            self.ps[t] = softmax(y)

            # calculate the loss 
            loss += -np.log(self.ps[t][sentence[t], 0]+1e-12)
        
        self.h_neg1 = self.hs[len(sentence)-1]

        return loss

    def _backward(self, target: List[int]):
        self.dwxh = self._zero_initialize(self.n, self.dic_len)
        self.dwhh = self._zero_initialize(self.n, self.n)
        self.dwyh = self._zero_initialize(self.dic_len, self.n)
        
        self.dbh = self._zero_initialize(self.n, 1)
        self.dby = self._zero_initialize(self.dic_len, 1)

        dh_next = self._zero_initialize(self.n, 1)
        for t in reversed(range(len(target))):
            # 65x1
            dy = self.ps[t].copy()
            dy[target[t]] -= 1

            # n x 65 @ 65x1 + n x 1
            dh = self.wyh.T @ dy + dh_next

            # 
            dtanh = (1 - self.hs[t] * self.hs[t]) * dh
                
            # n x 65 = n x 1 @ 1 x 65
            self.dwxh += dtanh @ self.xs[t].T
            self.dwhh += dtanh @ self.hs[t-1].T
            self.dwyh += dy @ self.hs[t].T

            self.dbh += dtanh
            self.dby += dy

            dh_next = self.whh.T @ dtanh
        
    def _update_parameters(self):

        # SGD a not good enough method to optimize
        # for grad in [self.dwxh, self.dwyh, self.dwhh, self.dbh, self.dby]:
            # np.clip(grad, -5, 5, out=grad)
            
        # self.wxh -= self.learning_rate * self.dwxh
        # self.wyh -= self.learning_rate * self.dwyh
        # self.whh -= self.learning_rate * self.dwhh
        
        # self.bh -= self.learning_rate * self.dbh
        # self.by -= self.learning_rate * self.dby

        # adagrad
        params = [self.wxh, self.whh, self.wyh, self.bh, self.by]
        grads = [self.dwxh, self.dwhh, self.dwyh, self.dbh, self.dby]
        mems = [self.mwxh, self.mwhh, self.mwyh, self.mbh, self.mby]

        for p, g, m in zip(params, grads, mems):
            np.clip(g, -5, 5, out=g)
            m += g * g
            p -= self.learning_rate * g / np.sqrt(m + 1e-8)
        
        # adam optimize

    def train(self):
        log_info('start training')
        for epoch in range(1, epochs+1):
            tot_loss = 0
            log_info(f'epoch: {epoch} started')
            for idx, (sentence, target, is_epoch_end) in enumerate(self.loader.get_sentences()):
                if is_epoch_end:
                    self.h_neg1 = self._zero_initialize(self.n, 1)
                    break
                
                # 1. forward
                loss = self._forward(sentence)

                # 2. tot loss
                tot_loss += loss

                # 3. backward
                self._backward(target)

                # 4. update the parameters
                self._update_parameters()

                # 5. print the average loss per 100 trains
                if idx % 100 == 0:
                    log_info(f'epoch {epoch} iteration {idx} average loss {loss/100:.4f}')

                # 6. sample
                if idx % 1000 == 0:
                    seed = sentence[0]
                    print(f'---- epoch {epoch} iteration {idx} test sample seed {self.loader.ix_to_char[seed]} ----')
                    sample_text = self.sample(seed, sample_len)
                    print(f'sample text: {sample_text}')
                    print('-' * 50)
                
                # 7. sample with prompt
                if idx % 5000 == 0:
                    prompt_raw = np.random.choice(self.prompts)
                    prompt = [self.loader.char_to_ix[c] for c in prompt_raw]
                    print(f'---- epoch {epoch} iteration {idx} test sample prompt "{prompt_raw}" ----')
                    sample_text = self.sample_with_prompt(prompt, sample_len)
                    print(f'sample text: {sample_text}')
                    print('-' * 50)

            # total loss of current epoch
            log_info(f'epoch {epoch} tot loss {tot_loss:.4f}')
    
    # return a sentence by seed
    def sample(self, seed_ix: int, cnt: int) -> str:
        # 65x1
        x = self._zero_initialize(self.dic_len, 1)

        x[seed_ix] = 1
        h = self.h_neg1.copy()

        indices = [seed_ix]
        for t in range(cnt):
            h = np.tanh(self.wxh @ x + self.whh @ h + self.bh)
            y = self.wyh @ h + self.by
            p = softmax(y)

            ix = np.random.choice(self.dic_len, p=p.ravel())

            x = self._zero_initialize(self.dic_len, 1)
            x[ix] = 1

            indices.append(ix)
        
        sentence = ''.join(self.loader.ix_to_char[ix] for ix in indices) 

        return sentence
    
    def sample_with_prompt(self, prompt: List[int], cnt: int) -> str:

        # get last state h
        h = self.h_neg1.copy()
        for ix in prompt:
            x = self._zero_initialize(self.dic_len, 1)
            x[ix] = 1

            h = np.tanh(self.wxh @ x + self.whh @ h + self.bh)
            
        indices = prompt.copy()

        y = self.wyh @ h + self.by
        p = softmax(y / 0.7)
        ix = np.random.choice(self.dic_len, p=p.ravel())
        
        x = self._zero_initialize(self.dic_len, 1)
        x[ix] = 1
        indices.append(ix)

        for t in range(cnt):
            h = np.tanh(self.wxh @ x + self.whh @ h + self.bh)
            y = self.wyh @ h + self.by
            p = softmax(y / 0.7)

            ix = np.random.choice(self.dic_len, p=p.ravel())

            x = self._zero_initialize(self.dic_len, 1)
            x[ix] = 1

            indices.append(ix)

        sentence = ''.join(self.loader.ix_to_char[ix] for ix in indices) 
        
        return sentence


def main():
    loader = DataLoader(os.path.join(file_path, TINY_SHAKESPEARE_FILENAME))
    
    tiny_shakespeare = MyShakespeare(loader)

    tiny_shakespeare.train()


if __name__ == '__main__':
    main()
