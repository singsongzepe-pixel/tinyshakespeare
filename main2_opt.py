import numpy as np
from typing import List, Dict, Tuple
import os

from constants import *
from dataloader import DataLoader
from utils import *

# GRU RNN + adam optimizer
class MyShakespeare:
    def __init__(self, loader: DataLoader):
        
        #
        self.loader = loader
        self.dic_len = self.loader.dic_len # 65 as value
        self.T = self.loader.sentence_len
        self.bs = self.loader.sentence_batch_size
        
        self.epochs = epochs

        # hyperparameter
        self.n = 512
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.lambda_ = lambda_

        # for sampling
        self.prompts = prompts

        # hidden parameters
        # the parameter between cells
        #  update gate
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

        # for adam optimizer
        self.rwiz = np.zeros_like(self.wiz)
        self.rwhz = np.zeros_like(self.whz)
        self.rbz = np.zeros_like(self.bz)
        self.rwir = np.zeros_like(self.wir)
        self.rwhr = np.zeros_like(self.whr)
        self.rbr = np.zeros_like(self.br)
        self.rwin = np.zeros_like(self.win)
        self.rwhn = np.zeros_like(self.whn)
        self.rbn = np.zeros_like(self.bn)
        self.rwyh = np.zeros_like(self.wyh)
        self.rby = np.zeros_like(self.by)
        
        self.vwiz = np.zeros_like(self.wiz)
        self.vwhz = np.zeros_like(self.whz)
        self.vbz = np.zeros_like(self.bz)
        self.vwir = np.zeros_like(self.wir)
        self.vwhr = np.zeros_like(self.whr)
        self.vbr = np.zeros_like(self.br)
        self.vwin = np.zeros_like(self.win)
        self.vwhn = np.zeros_like(self.whn)
        self.vbn = np.zeros_like(self.bn)
        self.vwyh = np.zeros_like(self.wyh)
        self.vby = np.zeros_like(self.by)

        # record how many times params updated
        self.iter_time = 0

        # store some state from forward, will be accessed in backward
        # Time Dimension Batch size 
        self.xs = np.zeros((self.T, self.dic_len, self.bs))
        self.zs = np.zeros((self.T, self.n, self.bs))
        self.rs = np.zeros((self.T, self.n, self.bs))
        self.hs_tilde = np.zeros((self.T, self.n, self.bs))
        self.hs = np.zeros((self.T, self.n, self.bs))
        self.ps = np.zeros((self.T, self.dic_len, self.bs))

        # store state of last sentence
        self.h_neg1 = self._zero_initialize(self.n, 1)

        # for adagrad
        self.params    = [self.wiz, self.whz, self.bz, self.wir, self.whr, self.br, self.win, self.whn, self.bn, self.wyh, self.by]
        self.grads     = [self.dwiz, self.dwhz, self.dbz, self.dwir, self.dwhr, self.dbr, self.dwin, self.dwhn, self.dbn, self.dwyh, self.dby]
        self.mems      = [self.rwiz, self.rwhz, self.rbz, self.rwir, self.rwhr, self.rbr, self.rwin, self.rwhn, self.rbn, self.rwyh, self.rby]
        self.momentums = [self.vwiz, self.vwhz, self.vbz, self.vwir, self.vwhr, self.vbr, self.vwin, self.vwhn, self.vbn, self.vwyh, self.vby]

        # total parameter count
        tot_params = self.wiz.size + self.whz.size + self.bz.size + \
                     self.wir.size + self.whr.size + self.br.size + \
                     self.win.size + self.whn.size + self.bn.size + \
                     self.wyh.size + self.by.size
        log_info(f'the RNN model has a total of {tot_params} parameters')

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

    def _forward(self, batch_sentences) -> float:
        bs = batch_sentences.shape[0]
        T = self.T

        h_prev = np.tile(self.h_neg1, (1, bs))
        self.hs_minus_1 = h_prev.copy()

        loss = 0
        for t in range(T):
            char_indices = batch_sentences[:, t] # (bs,)
            
            self.xs[t, char_indices, np.arange(bs)] = 1

            wiz_x = self.wiz[:, char_indices]
            wir_x = self.wir[:, char_indices]
            win_x = self.win[:, char_indices]

            # (n, n) @ (n, bs) -> (n, bs)
            self.zs[t] = sigmoid(wiz_x + self.whz @ h_prev + self.bz)
            self.rs[t] = sigmoid(wir_x + self.whr @ h_prev + self.br)
            
            # Candidate hidden state
            self.hs_tilde[t] = np.tanh(win_x + self.whn @ (self.rs[t] * h_prev) + self.bn)

            # Hidden state update
            self.hs[t] = (1 - self.zs[t]) * h_prev + self.zs[t] * self.hs_tilde[t]
            
            # Output & Softmax
            y = self.wyh @ self.hs[t] + self.by # (dic_len, bs)
            self.ps[t] = softmax(y)

            corect_char_probs = self.ps[t][char_indices, np.arange(bs)]
            loss += -np.sum(np.log(corect_char_probs + 1e-12))
            
            h_prev = self.hs[t]

        self.h_neg1 = np.mean(self.hs[T-1], axis=1, keepdims=True)

        return loss / bs

    def _backward(self, batch_targets):
        bs = batch_targets.shape[0]
        T = self.T
        
        for g in self.grads:
            g.fill(0)
        
        dh_next = np.zeros((self.n, bs))
        
        for t in reversed(range(T)):
            char_indices = batch_targets[:, t]
            
            # (dic_len, bs)
            dy = self.ps[t].copy()
            dy[char_indices, np.arange(bs)] -= 1
            
            self.dwyh += dy @ self.hs[t].T
            self.dby += np.sum(dy, axis=1, keepdims=True)

            dh = self.wyh.T @ dy + dh_next

            h_prev = self.hs[t-1] if t > 0 else self.hs_minus_1

            dz_raw = dh * (self.hs_tilde[t] - h_prev) * self.zs[t] * (1 - self.zs[t])
            dn_raw = dh * self.zs[t] * (1 - self.hs_tilde[t]**2)
            dr_raw = (self.whn.T @ dn_raw) * h_prev * self.rs[t] * (1 - self.rs[t])

            np.add.at(self.dwiz.T, char_indices, dz_raw.T)
            np.add.at(self.dwir.T, char_indices, dr_raw.T)
            np.add.at(self.dwin.T, char_indices, dn_raw.T)

            self.dwhz += dz_raw @ h_prev.T
            self.dwhr += dr_raw @ h_prev.T
            self.dwhn += dn_raw @ (self.rs[t] * h_prev).T

            # bais
            self.dbz += np.sum(dz_raw, axis=1, keepdims=True)
            self.dbr += np.sum(dr_raw, axis=1, keepdims=True)
            self.dbn += np.sum(dn_raw, axis=1, keepdims=True)

            # ! pass loss to the cell before current cell
            dh_next = dh * (1 - self.zs[t]) + \
                    (self.whz.T @ dz_raw) + \
                    (self.whr.T @ dr_raw) + \
                    (self.whn.T @ dn_raw) * self.rs[t]

    def _update_parameters(self):
        # adam optimize
        self.iter_time += 1

        for p, g, r, v in zip(self.params, self.grads, self.mems, self.momentums):
            np.clip(g, -5, 5, out=g)

            v[:] = self.alpha * v + (1-self.alpha) * g

            r[:] = self.lambda_ * r + (1-self.lambda_) * (g * g)

            v_hat = v / (1-self.alpha ** self.iter_time)
            r_hat = r / (1-self.lambda_ ** self.iter_time)

            p -= self.learning_rate * v_hat / np.sqrt(r_hat + epsilon)

    def train(self):
        log_info('start batch training')
        for epoch in range(1, epochs+1):
            tot_loss = 0
            log_info(f'epoch: {epoch} started')
            for idx, (sentence, target, is_epoch_end) in enumerate(self.loader.get_sentences_batch()):
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
                if idx % 10 == 0:
                    log_info(f'epoch {epoch} iteration {idx} average loss {loss/100:.4f}')

                # 6. sample
                if idx % 50 == 0:
                    # as sentence is a 2d-array
                    seed = sentence[0, 0]
                    print(f'---- epoch {epoch} iteration {idx} test sample seed {self.loader.ix_to_char[seed]} ----')
                    sample_text = self.sample(seed, sample_len)
                    print(f'sample text: {sample_text}')
                    print('-' * 50)
                
                # 7. sample with prompt
                if idx % 250 == 0:
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

    # if not tiny_shakespeare.load_model():
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
