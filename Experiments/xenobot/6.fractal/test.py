import numpy as np

def toBinary(number, bit):
    return ((number & (1<<np.arange(bit))) > 0).astype(int)

w = np.arange(8).reshape([2,2,2])
w = w.reshape([2,-1])
w = w.reshape([2,2,2])
print(w)