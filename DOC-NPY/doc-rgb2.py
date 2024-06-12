import numpy as np
import pandas as pd

# Đọc file .npy với allow_pickle=True
data = np.load('DACTRUNG/rgb.npy', allow_pickle=True)

# Chuyển đổi dữ liệu thành một danh sách các từ điển
data_list = [{'image_name': item['image_name'], **{f'rgb_feature_{idx}': value for idx, value in enumerate(item['rgb_features'])}} for item in data]

# Tạo DataFrame từ danh sách các từ điển
df = pd.DataFrame(data_list)

# Lưu DataFrame vào file CSV
df.to_csv('DOC-NPY/rgb_features2.csv', index=False)

print("Features saved to CSV successfully!")
