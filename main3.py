import numpy as np
from typing import *
import os

from constants import *
from tokenized_dataloader import TokenDataLoader
from utils import *

class TinyShakespeare:
    def __init__(self):
        
        # data provider
        self.loader = TokenDataLoader()

        self.epochs = epochs

        # hyperparameters
        self.n = 512                           # hidden layer cell count
        self.embed_dim = 128                   # mapping dim
        self.word_cnt = self.loader.word_cnt   # total deduplicate word count     
        self.learning_rate = 0.001
        self.alpha = 0.99
        self.lambda_ = 0.999

        # sample
        self.sample_len = 100
        self.sample_with_prompt_len = 150

        # hidden parameters
        # the parameter between cells
        # embedding layer (D, W)
        self.wembed = self._xavier_initialize(self.embed_dim, self.word_cnt)
        self.dwembed = np.zeros_like(self.wembed)
        #  update gate
        self.wiz  = self._xavier_initialize(self.n, self.embed_dim)
        self.dwiz = np.zeros_like(self.wiz)
        self.whz  = self._xavier_initialize(self.n, self.n)
        self.dwhz = np.zeros_like(self.whz)
        self.bz   = self._zero_initialize(self.n, 1)
        self.dbz  = np.zeros_like(self.bz)
        # reset gate
        self.wir  = self._xavier_initialize(self.n, self.embed_dim)
        self.dwir = np.zeros_like(self.wir)
        self.whr  = self._xavier_initialize(self.n, self.n)
        self.dwhr = np.zeros_like(self.whr)
        self.br   = self._zero_initialize(self.n, 1)
        self.dbr  = np.zeros_like(self.br)
        # new / candidate
        self.win  = self._xavier_initialize(self.n, self.embed_dim)
        self.dwin = np.zeros_like(self.win)
        self.whn  = self._xavier_initialize(self.n, self.n)
        self.dwhn = np.zeros_like(self.whn)
        self.bn   = self._zero_initialize(self.n, 1)
        self.dbn  = np.zeros_like(self.bn)

        # the parameter of output
        self.wyh  = self._xavier_initialize(self.word_cnt, self.n)
        self.dwyh = np.zeros_like(self.wyh)
        self.by   = self._zero_initialize(self.word_cnt, 1)
        self.dby  = np.zeros_like(self.by)

        # for adam optimizer
        self.rwembed = np.zeros_like(self.wembed)
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
        
        self.vwembed = np.zeros_like(self.wembed)
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
        self.xs = None
        self.zs = None
        self.rs = None
        self.ps = None
        self.hs = None
        self.hs_tilde = None

        # store state of last sentence
        self.h_neg1 = self._zero_initialize(self.n, 1)

        # for adagrad
        self.params    = [self.wembed, self.wiz, self.whz, self.bz, self.wir, self.whr, self.br, self.win, self.whn, self.bn, self.wyh, self.by]
        self.grads     = [self.dwembed, self.dwiz, self.dwhz, self.dbz, self.dwir, self.dwhr, self.dbr, self.dwin, self.dwhn, self.dbn, self.dwyh, self.dby]
        self.mems      = [self.rwembed, self.rwiz, self.rwhz, self.rbz, self.rwir, self.rwhr, self.rbr, self.rwin, self.rwhn, self.rbn, self.rwyh, self.rby]
        self.momentums = [self.vwembed, self.vwiz, self.vwhz, self.vbz, self.vwir, self.vwhr, self.vbr, self.vwin, self.vwhn, self.vbn, self.vwyh, self.vby]

        # total parameter count
        tot_params = self.wembed.size + \
                     self.wiz.size + self.whz.size + self.bz.size + \
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

    def _forward(self, seq_ix: List[int]):
        self.xs, self.zs, self.rs = {}, {}, {}
        self.ps, self.hs, self.hs_tilde = {}, {}, {}

        self.hs[-1] = self.h_neg1.copy()
        # t = [0, 64) 
        for t in range(self.loader.seq_len):
            # token ix
            ix = seq_ix[t]
            
            # get corresponding token vector
            self.xs[t] = self.wembed[:, [ix]]

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
        
        self.h_neg1 = self.hs[self.loader.seq_len - 1].copy()

    def _backward(self, seq_ix: List[int]):
        for g in self.grads:
            g.fill(0)
        
        dh_next = self._zero_initialize(self.n, 1)
        # t = [64...1]
        for t in reversed(range(self.loader.seq_len)):
            # 
            dy = self.ps[t].copy()
            dy[seq_ix[t+1]] -= 1
            
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
            
            # embedding layer
            ix = seq_ix[t] # previous token
            dx_t = self.wiz.T @ dz_raw + \
                   self.wir.T @ dr_raw + \
                   self.win.T @ dn_raw
            self.dwembed[:, [ix]] += dx_t

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
            for idx, (seq_ix, is_epoch_end) in enumerate(self.loader.get_tokens()):
                if is_epoch_end:
                    self.h_neg1 = self._zero_initialize(self.n, 1)
                    break

                # 1. forward
                self._forward(seq_ix)

                # 2. tot loss
                loss = 0
                for t in range(self.loader.seq_len):
                    loss += -np.log(self.ps[t][seq_ix[t+1], 0] + 1e-12)
                tot_loss += loss

                # 3. backward
                self._backward(seq_ix)

                # 4. update the parameters
                self._update_parameters()

                # 5. print the average loss per 100 trains
                if idx % 20 == 0:
                    log_info(f'epoch {epoch} iteration {idx} average loss {loss/self.loader.seq_len:.4f}')

                # 6. sample
                if idx % 200 == 0:
                    seed_ix = seq_ix[0]
                    print(f'---- epoch {epoch} iteration {idx} test sample seed {self.loader.ix_to_word[seed_ix]} ----')
                    sample_text = self.sample_no_state(seed_ix, self.sample_len)
                    print(f'sample text: {sample_text}')
                    print('-' * 50)
                
                # 7. sample with prompt
                # if idx % 5000 == 0:
                #     prompt_raw = np.random.choice(self.prompts)
                #     prompt = [self.loader.char_to_ix[c] for c in prompt_raw]
                #     print(f'---- epoch {epoch} iteration {idx} test sample prompt "{prompt_raw}" ----')
                #     sample_text = self.sample_with_prompt(prompt, sample_len)
                #     print(f'sample text: {sample_text}')
                #     print('-' * 50)

            # total loss of current epoch
            log_info(f'epoch {epoch} tot loss {tot_loss:.4f}')

    def sample_no_state(self, seed_ix: int, cnt: int) -> str:
        return self.sample(seed_ix, cnt, self._zero_initialize(self.n, 1))
    
    def sample_with_state(self, seed_ix: int, cnt: int, h: np.array) -> str:
        return self.sample(seed_ix, cnt, h)

    def sample(self, seed_ix: int, cnt: int, h_: np.array) -> str:
        x = self.wembed[:, [seed_ix]].copy()
        h = h_.copy()

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

            ix = np.random.choice(self.word_cnt, p=p.ravel() / np.sum(p))

            x = self.wembed[:, [ix]].copy()

            indices.append(ix)
        
        sentence = ' '.join(self.loader.ix_to_word[idx] for idx in indices) 

        return sentence

    def sample_with_prompt(self, prompt: List[int], cnt: int) -> str:

        pass

    def save_model(self, filename = MODEL_PARAMS_FILE_PATH + GRU_RNN_WITH_ADAM_EMBEDDING_PARAMS_FILENAME):
        _dir = os.path.dirname(filename)
        if not os.path.exists(_dir):
            os.mkdir(_dir)

        if os.path.exists(filename):
            log_info(f'model params file already exist, trying to overwrite it')
        
        np.savez(filename,
                 wembed=self.wembed,
                 wiz=self.wiz, whz=self.whz, bz=self.bz,
                 wir=self.wir, whr=self.whr, br=self.br,
                 win=self.win, whn=self.whn, bn=self.bn,
                 wyh=self.wyh, by=self.by)
        
        log_info(f'model params file save at {filename}')

    def load_model(self, filename = MODEL_PARAMS_FILE_PATH + GRU_RNN_WITH_ADAM_EMBEDDING_PARAMS_FILENAME) -> bool:
        if not os.path.exists(filename):
            log_info(f'model params file doesn\'t exist: {filename}')
            return False

        data = np.load(filename)
        self.wembed = data['wembed']
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
    ts = TinyShakespeare()

    if not ts.load_model():
        ts.train()
        ts.save_model()
    
    sample_txt = ts.sample_no_state(ts.loader.word_to_ix['thou'], 200)
    print(sample_txt)

if __name__ == '__main__':
    main()
