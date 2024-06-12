import cv2
import numpy as np
import os

images_dir = 'DATA/'
image_files = os.listdir(images_dir)
rgb_features_list = []

def convert_to_rgb(img_path):
    image = cv2.imread(img_path)
    hist_rgb = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_rgb, hist_rgb)
    return hist_rgb.flatten()

# Sử dụng Color Histogram (RGB) để trích xuất đặc trưng cho từng ảnh
dem = 0
for img_file in image_files:
    img_path = os.path.join(images_dir, img_file)
    rgb_data = convert_to_rgb(img_path)
    print(rgb_data.shape)
    rgb_features_list.append({'image_name': img_file, 'rgb_features': rgb_data})
    print(f"Trích rút thành công {img_file}! \n")
    dem = dem + 1

np.save("DACTRUNG/rgball.npy", rgb_features_list)
