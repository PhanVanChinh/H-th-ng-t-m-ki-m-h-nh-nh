import cv2
import numpy as np
import pandas as pd
import os

images_dir = 'DATA/'
image_files = os.listdir(images_dir)
rgb_features_list = []
block_size = 64

def convert_to_rgb_blocks(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    rgb_features = np.array([])
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y+block_size, x:x+block_size]
            hist_red = cv2.calcHist([block], [0], None, [16], [0, 256])
            hist_green = cv2.calcHist([block], [1], None, [16], [0, 256])
            hist_blue = cv2.calcHist([block], [2], None, [16], [0, 256])
            cv2.normalize(hist_red, hist_red)
            cv2.normalize(hist_green, hist_green)
            cv2.normalize(hist_blue, hist_blue)
            hist_rgb = np.concatenate((hist_red.flatten(), hist_green.flatten(), hist_blue.flatten()))
            rgb_features = np.concatenate((rgb_features, hist_rgb))
    return rgb_features

# Sử dụng Color Histogram (RGB) để trích xuất đặc trưng cho từng ảnh
for img_file in image_files:
    img_path = os.path.join(images_dir, img_file)
    rgb_data = convert_to_rgb_blocks(img_path)
    print(rgb_data.shape)
    print("Trích rút thành công cho:", img_file)
    rgb_features_list.append({'image_name': img_file, 'rgb_features': rgb_data})

np.save("DACTRUNG/rgb.npy", rgb_features_list)
