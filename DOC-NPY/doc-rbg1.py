import numpy as np
import pandas as pd

# Đọc file .npy với allow_pickle=True
data = np.load('DACTRUNG/rgb.npy', allow_pickle=True)

# Mở file CSV để ghi
with open('DOC-NPY/rgb_features1.csv', 'w') as f:
    # Ghi tiêu đề cột
    f.write("image_name,rgb_features\n")
    
    # Lặp qua từng item và ghi vào file CSV
    for item in data:
        image_name = item['image_name']
        rgb_features = ",".join(str(value) for value in item['rgb_features'])
        f.write(f"{image_name},{rgb_features}\n")

print("Features saved to CSV successfully!")
