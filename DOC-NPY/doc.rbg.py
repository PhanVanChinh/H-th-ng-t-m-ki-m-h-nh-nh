import numpy as np
import pandas as pd

# Đọc file .npy với allow_pickle=True
data = np.load('DACTRUNG/rgb.npy', allow_pickle=True)

# Chuyển đổi dữ liệu thành một danh sách các từ điển
data_list = [{'image_name': item['image_name'], 'rgb_features': item['rgb_features']} for item in data]

# Tạo DataFrame từ danh sách các từ điển
df = pd.DataFrame(data_list)

# Lưu DataFrame vào file CSV
with open('DOC-NPY/rgb_features.csv', 'w') as file:
    file.write(df.to_string(index=False))

print("Features saved to CSV successfully!")
