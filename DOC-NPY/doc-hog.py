import numpy as np
# Đọc file .npy với allow_pickle=True
data = np.load('DACTRUNG/hog.npy', allow_pickle=True)

for item in data:
    print(item)
