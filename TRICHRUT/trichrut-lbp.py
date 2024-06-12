import cv2
import numpy as np
import os

# Thư mục chứa các ảnh
images_dir = 'DATA/'
# Danh sách tên các file ảnh
image_files = os.listdir(images_dir)

lbp_features_list = []
# Định nghĩa hàm để tính toán đặc trưng LBP từ một ảnh xám
def convert_to_lbp(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Khởi tạo một ma trận LBP có kích thước bằng kích thước ảnh và giá trị ban đầu là 0
    lbp_image = np.zeros_like(image)
    height, width = image.shape

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Lấy ngưỡng của pixel trung tâm
            center = image[i, j]
            
            # So sánh giá trị của các pixel lân cận với ngưỡng của pixel trung tâm và gán giá trị cho LBP
            lbp_code = 0
            lbp_code |= (image[i-1, j-1] >= center) << 7
            lbp_code |= (image[i-1, j] >= center) << 6
            lbp_code |= (image[i-1, j+1] >= center) << 5
            lbp_code |= (image[i, j+1] >= center) << 4
            lbp_code |= (image[i+1, j+1] >= center) << 3
            lbp_code |= (image[i+1, j] >= center) << 2
            lbp_code |= (image[i+1, j-1] >= center) << 1
            lbp_code |= (image[i, j-1] >= center) << 0
            # Gán giá trị LBP cho pixel tương ứng
            lbp_image[i, j] = lbp_code
    # Tính toán histogram của LBP
    hist, _ = np.histogram(lbp_image, bins=256, range=(0, 256))
    # Chuẩn hóa histogram
    hist = hist.astype(float) / np.sum(hist)
    return hist.flatten()

# Thực hiện trích xuất đặc trưng HOG cho từng ảnh
for i,img_file in enumerate(image_files): 
    img_path = os.path.join(images_dir, img_file)

    lbp_data = convert_to_lbp(img_path)
    lbp_features_list.append({'image_name': img_file, 'lbp_features': lbp_data})
    print(lbp_data.shape, end="\n")
    print(f'{i} Trich rut thanh cong!', end='\n')
print('\n')
np.save("DACTRUNG/lbp.npy", lbp_features_list)




