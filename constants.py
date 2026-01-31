
# singsongzepe
SINGSONGZEPE_ = 'SingSongZepe'
SINGSONGLOG_ = 'SingSongLog '
INFO = '[INFO]: '
WARN = '[WARN]: '
ERRO = '[ERRO]: '

DEBUG = True
LOG = True

file_path = './data/'

TINY_SHAKESPEARE_FILENAME = 'tiny_shakespeare.txt'
TINY_SHAKESPEARE_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

# model
# the hidden size, memory capacity
n = 128
sentence_len = 100
sample_len = 200
epochs = 10

learning_rate = 0.1

# for adam optimizer
alpha = 0.9
lambda_ = 0.999
epsilon = 1e-8

# for batch training
sentence_batch_size = 64

prompts = [
    "ROMEO: I take thee at thy word. Call me but love, and I'll be new",
    "Enter KING LEAR and",
    "KING: Our state of Denmark, together with the",
    "CLOWN: Truly, sir, the better for my"
]

# for saving model
MODEL_PARAMS_FILE_PATH = './model/'
MODEL_PARAMS_FILENAME = 'params.npz'
