import numpy as np

from constants import *

# debug/log function
def debug_info(*mesages):
    if DEBUG:
        print(SINGSONGLOG_ + INFO + ' '.join(mesages))

def debug_warn(*mesages):
    if DEBUG:
        print(SINGSONGLOG_ + WARN + ' '.join(mesages))

def debug_erro(*mesages):
    if DEBUG:
        print(SINGSONGLOG_ + ERRO + ' '.join(mesages))

def log_info(*mesages):
    if LOG:
        print(SINGSONGLOG_ + INFO + ' '.join(mesages))

def log_warn(*mesages):
    if LOG:
        print(SINGSONGLOG_ + WARN + ' '.join(mesages))

def log_erro(*mesages):
    if LOG:
        print(SINGSONGLOG_ + ERRO + ' '.join(mesages))


# for model
def softmax(z):
    tmp = np.exp(z - np.max(z, axis=0, keepdims=True))
    return tmp / np.sum(tmp, axis=0, keepdims=True)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
