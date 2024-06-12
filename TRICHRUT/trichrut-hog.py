import cv2
import numpy as np
from skimage.feature import hog
import os

# Thư mục chứa các ảnh
images_dir = 'DATA/'
# Danh sách tên các file ảnh
image_files = os.listdir(images_dir)

# Danh sách chứa đặc trưng HOG và định danh của từng ảnh
hog_features_list = []
def convert_to_hog(img_path):
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (fd, hog_image) = hog(gray_image, orientations=9, pixels_per_cell=(32, 32), 
                          cells_per_block=(2, 2), visualize=True, block_norm='L2')
    return fd

# Thực hiện trích xuất đặc trưng HOG cho từng ảnh
for i,img_file in enumerate(image_files): 
    img_path = os.path.join(images_dir, img_file)
    hog_data = convert_to_hog(img_path)
    hog_features_list.append({'image_name': img_file, 'hog_features': hog_data})
    print(hog_data.shape, end="\n")
    print(f'{i} Trich rut thanh cong!', end='\n')
print('\n')
np.save("DACTRUNG/hog.npy", hog_features_list)




