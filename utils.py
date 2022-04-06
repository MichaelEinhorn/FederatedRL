import numpy as np
import time


def softmax(x, axis=-1):
    if axis == -1:
        x = np.exp(x - np.max(x))
        return x / np.sum(x)
    x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    # x = np.exp(x)
    # print(x)
    return x / np.sum(x, axis=axis, keepdims=True)


def getArgs(*args, **kwargs):
    all_args = {("arg" + str(idx + 1)): arg for idx, arg in enumerate(args)}
    all_args.update(kwargs)
    return all_args
