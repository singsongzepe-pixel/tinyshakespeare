import numpy as np
from typing import List, Dict, Tuple
import os

from constants import *
from dataloader import DataLoader
from utils import *

# GRU RNN + adagrad
class MyShakespeare:
    def __init__(self, loader: DataLoader):
        
        #
        self.loader = loader
        self.dic_len = self.loader.dic_len # 65 as value
        
        self.epochs = epochs

        # hyperparameter
        self.n = 512
        self.learning_rate = learning_rate

        # for sampling
        self.prompts = prompts

        # hidden parameters
        # the parameter between cells
        # update gate
        self.wiz  = self._xavier_initialize(self.n, self.dic_len)
        self.dwiz = np.zeros_like(self.wiz)
        self.whz  = self._xavier_initialize(self.n, self.n)
        self.dwhz = np.zeros_like(self.whz)
        self.bz   = self._zero_initialize(self.n, 1)
        self.dbz  = np.zeros_like(self.bz)
        # reset gate
        self.wir  = self._xavier_initialize(self.n, self.dic_len)
        self.dwir = np.zeros_like(self.wir)
        self.whr  = self._xavier_initialize(self.n, self.n)
        self.dwhr = np.zeros_like(self.whr)
        self.br   = self._zero_initialize(self.n, 1)
        self.dbr  = np.zeros_like(self.br)
        # new / candidate
        self.win  = self._xavier_initialize(self.n, self.dic_len)
        self.dwin = np.zeros_like(self.win)
        self.whn  = self._xavier_initialize(self.n, self.n)
        self.dwhn = np.zeros_like(self.whn)
        self.bn   = self._zero_initialize(self.n, 1)
        self.dbn  = np.zeros_like(self.bn)

        # the parameter of output
        self.wyh  = self._xavier_initialize(self.dic_len, self.n)
        self.dwyh = np.zeros_like(self.wyh)
        self.by   = self._zero_initialize(self.dic_len, 1)
        self.dby  = np.zeros_like(self.by)

        # for adagrad
        self.mwiz = np.zeros_like(self.wiz)
        self.mwhz = np.zeros_like(self.whz)
        self.mbz = np.zeros_like(self.bz)
        self.mwir = np.zeros_like(self.wir)
        self.mwhr = np.zeros_like(self.whr)
        self.mbr = np.zeros_like(self.br)
        self.mwin = np.zeros_like(self.win)
        self.mwhn = np.zeros_like(self.whn)
        self.mbn = np.zeros_like(self.bn)
        self.mwyh = np.zeros_like(self.wyh)
        self.mby = np.zeros_like(self.by)

        # store some state from forward, will be accessed in backward
        self.xs = None
        self.zs = None
        self.rs = None
        self.ps = None
        self.hs = None
        self.hs_tilde = None

        # store state of last sentence
        self.h_neg1 = self._zero_initialize(self.n, 1)

        # for optimizing
        self.params = [self.wiz, self.whz, self.bz, self.wir, self.whr, self.br, self.win, self.whn, self.bn, self.wyh, self.by]
        self.grads  = [self.dwiz, self.dwhz, self.dbz, self.dwir, self.dwhr, self.dbr, self.dwin, self.dwhn, self.dbn, self.dwyh, self.dby]
        # for adagrad
        self.mems   = [self.mwiz, self.mwhz, self.mbz, self.mwir, self.mwhr, self.mbr, self.mwin, self.mwhn, self.mbn, self.mwyh, self.mby]

        # total parameter count
        tot_params = self.wiz.size + self.whz.size + self.bz.size + \
                     self.wir.size + self.whr.size + self.br.size + \
                     self.win.size + self.whn.size + self.bn.size + \
                     self.wyh.size + self.by.size
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
        oh = self._zero_initialize(self.dic_len, 1)
        oh[ix] = 1

        return oh

    def _forward(self, sentence) -> float:
        self.xs, self.zs, self.rs = {}, {}, {} 
        self.ps, self.hs, self.hs_tilde = {}, {}, {}

        self.hs[-1] = self.h_neg1.copy()
        loss = 0
        for t in range(self.loader.sentence_len):
            # (65,1)
            self.xs[t] = self.one_hot(sentence[t])
            
            # update gate
            self.zs[t] = sigmoid(self.wiz @ self.xs[t] + self.whz @ self.hs[t-1] + self.bz)

            # reset gate
            self.rs[t] = sigmoid(self.wir @ self.xs[t] + self.whr @ self.hs[t-1] + self.br)

            # candidate
            self.hs_tilde[t] = np.tanh(self.win @ self.xs[t] + self.whn @ (self.rs[t] * self.hs[t-1]) + self.bn)

            # h
            self.hs[t] = (1-self.zs[t]) * self.hs[t-1] + self.zs[t] * self.hs_tilde[t]
            y = self.wyh @ self.hs[t] + self.by
            self.ps[t] = softmax(y)

            # loss
            loss += -np.log(self.ps[t][sentence[t], 0] + 1e-12)
        
        self.h_neg1 = self.hs[self.loader.sentence_len - 1].copy()

        return loss

    def _backward(self, target: List[int]):
        for g in self.grads:
            g.fill(0)
        
        dh_next = self._zero_initialize(self.n, 1)
        for t in reversed(range(self.loader.sentence_len)):
            # 
            dy = self.ps[t].copy()
            dy[target[t]] -= 1
            
            self.dwyh += dy @ self.hs[t].T
            self.dby += dy

            dh = self.wyh.T @ dy + dh_next

            # update gate
            dz_raw = dh * (self.hs_tilde[t] - self.hs[t-1]) * self.zs[t] * (1 - self.zs[t])
            self.dwiz += dz_raw @ self.xs[t].T
            self.dwhz += dz_raw @ self.hs[t-1].T
            self.dbz  += dz_raw

            # reset gate
            dn_raw = dh * self.zs[t] * (1 - self.hs_tilde[t]**2)
            dr_raw = (self.whn.T @ dn_raw) * self.hs[t-1] * self.rs[t] * (1 - self.rs[t])
            self.dwir += dr_raw @ self.xs[t].T
            self.dwhr += dr_raw @ self.hs[t-1].T
            self.dbr  += dr_raw

            # new/candidate
            self.dwin += dn_raw @ self.xs[t].T
            self.dwhn += dn_raw @ (self.rs[t] * self.hs[t-1]).T
            self.dbn  += dn_raw

            # ! pass loss to the cell before current cell
            dh_next = dh * (1 - self.zs[t]) + \
                      (self.whz.T @ dz_raw) + \
                      (self.whr.T @ dr_raw) + \
                      (self.whn.T @ dn_raw) * self.rs[t]

        
    def _update_parameters(self):
        # adagrad
        for p, g, m in zip(self.params, self.grads, self.mems):
            np.clip(g, -5, 5, out=g)
            m += g * g
            p -= self.learning_rate * g / np.sqrt(m + 1e-8)
        
        # adam optimize

    def train(self):
        log_info('start batch training')
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
            # update gate
            z = sigmoid(self.wiz @ x + self.whz @ h + self.bz)

            # reset gate
            r = sigmoid(self.wir @ x + self.whr @ h + self.br)

            # candidate
            h_tilde = np.tanh(self.win @ x + self.whn @ (r * h) + self.bn)

            # h
            h = (1 - z) * h + z * h_tilde
            
            y = self.wyh @ h + self.by
            p = softmax(y / 0.7)

            ix = np.random.choice(self.dic_len, p=p.ravel())

            x = self._zero_initialize(self.dic_len, 1)
            x[ix] = 1

            indices.append(ix)
        
        sentence = ''.join(self.loader.ix_to_char[idx] for idx in indices) 

        return sentence
    
    def sample_with_prompt(self, prompt: List[int], cnt: int) -> str:

        # get last state h
        h = self.h_neg1.copy()
        indices = prompt.copy()

        for i in range(len(prompt) - 1):
            x = self._zero_initialize(self.dic_len, 1)
            ix = prompt[i]
            x[ix] = 1

            # update gate
            z = sigmoid(self.wiz @ x + self.whz @ h + self.bz)
            # reset gate
            r = sigmoid(self.wir @ x + self.whr @ h + self.br)
            # candidate
            h_tilde = np.tanh(self.win @ x + self.whn @ (r * h) + self.bn)
            # h
            h = (1 - z) * h + z * h_tilde

        x = self._zero_initialize(self.dic_len, 1)
        x[prompt[-1]] = 1

        for t in range(cnt):
            # update gate
            z = sigmoid(self.wiz @ x + self.whz @ h + self.bz)

            # reset gate
            r = sigmoid(self.wir @ x + self.whr @ h + self.br)

            # candidate
            h_tilde = np.tanh(self.win @ x + self.whn @ (r * h) + self.bn)

            # h
            h = (1 - z) * h + z * h_tilde
            
            y = self.wyh @ h + self.by
            p = softmax(y / 0.7)

            ix = np.random.choice(self.dic_len, p=p.ravel())

            x = self._zero_initialize(self.dic_len, 1)
            x[ix] = 1

            indices.append(ix)

        sentence = ''.join(self.loader.ix_to_char[idx] for idx in indices) 
        
        return sentence

    def save_model(self, filename = MODEL_PARAMS_FILE_PATH + MODEL_PARAMS_FILENAME):
        _dir = os.path.dirname(filename)
        if not os.path.exists(_dir):
            os.mkdir(_dir)

        if os.path.exists(filename):
            log_info(f'model params file already exist, trying to overwrite it')
        
        np.savez(filename,
                 wiz=self.wiz, whz=self.whz, bz=self.bz,
                 wir=self.wir, whr=self.whr, br=self.br,
                 win=self.win, whn=self.whn, bn=self.bn,
                 wyh=self.wyh, by=self.by)
        
        log_info(f'model params file save at {filename}')

    def load_model(self, filename = MODEL_PARAMS_FILE_PATH + MODEL_PARAMS_FILENAME) -> bool:
        if not os.path.exists(filename):
            log_info(f'model params file doesn\'t exist: {filename}')
            return False

        data = np.load(filename)
        self.wiz = data['wiz']
        self.whz = data['whz']
        self.bz  = data['bz']
        self.wir = data['wir']
        self.whr = data['whr']
        self.br  = data['br']
        self.win = data['win']
        self.whn = data['whn']
        self.bn  = data['bn']
        self.wyh = data['wyh']
        self.by  = data['by']

        log_info(f'model params loaded successfully')
        return True

def main():
    loader = DataLoader(os.path.join(file_path, TINY_SHAKESPEARE_FILENAME))
    
    tiny_shakespeare = MyShakespeare(loader)

    if not tiny_shakespeare.load_model():
        tiny_shakespeare.train()
        tiny_shakespeare.save_model()

    # if well trained, test some samples
    for i in range(5):
        seed_ix = np.random.choice(tiny_shakespeare.dic_len)
        print(f'---- test sample seed {tiny_shakespeare.loader.ix_to_char[seed_ix]} ----')
        sample_text = tiny_shakespeare.sample(seed_ix, sample_len)
        print(f'sample text: {sample_text}')
        print('-' * 50)

    # and then test some samples with prompt
    for i in range(5):
        prompt_raw = np.random.choice(prompts)
        prompt = [tiny_shakespeare.loader.char_to_ix[c] for c in prompt_raw]
        print(f'---- test sample prompt "{prompt_raw}" ----')
        sample_text = tiny_shakespeare.sample_with_prompt(prompt, sample_len)
        print(f'sample text: {sample_text}')
        print('-' * 50)



if __name__ == '__main__':
    main()
