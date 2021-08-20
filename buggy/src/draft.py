import numpy as np

p1 = "/home/tim/SW/data/buggy/parking/buggy_JFC_timestamp.npy"
p2 = "/home/tim/SW/data/buggy/2021_08_20/buggy_BSU_timestamp.npy"
data = np.load(p2)
print(len(data))