import numpy as np

# Đọc file .npy với allow_pickle=True
data = np.load('DACTRUNG/lbp.npy', allow_pickle=True)

# Duyệt qua từng phần tử và in ra mỗi phần tử một dòng
for item in data:
    print(item)
