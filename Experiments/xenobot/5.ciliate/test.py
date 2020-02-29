import numpy as np

def toBinary(number, bit):
    return ((number & (1<<np.arange(bit))) > 0).astype(int)

for i in range(256):
    print(toBinary(i,8))