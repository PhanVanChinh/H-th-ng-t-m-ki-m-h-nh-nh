import cv2
import numpy as np
from skimage.feature import hog
import os
import matplotlib.pyplot as plt

# Thư mục chứa các ảnh
images_dir = 'DATA/'
# Danh sách tên các file ảnh
image_files = os.listdir(images_dir)

def convert_to_hog(img_path):
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (fd, hog_image) = hog(gray_image, orientations=9, pixels_per_cell=(32, 32), 
                          cells_per_block=(2, 2), visualize=True, block_norm='L2')
    return fd

def convert_to_rgb(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    block_size = 64
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

def convert_to_rgb1(img_path):
    image = cv2.imread(img_path)
    hist_rgb = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_rgb, hist_rgb)
    return hist_rgb.flatten()


def khoangcach_euclidean(x, y):
    squared_distance = 0
    for i in range(len(x)):
        squared_distance += (x[i] - y[i]) ** 2
    return squared_distance ** 0.5

def chuan_hoa_euclidean_ve_khoang(distance):
    #Chuẩn hóa khoảng cách euclidean về đoạn [0,1] để dễ tính toán (dùng công thức Min-Max)
    max_distance = np.max(distance)
    min_distance = np.min(distance)
    chuanhoa_distances = (distance-min_distance)/(max_distance - min_distance)
    return chuanhoa_distances

def danhsach_euclidean(input_image_dt, dt_features_list):
    #Hàm chạy tính khoảng cách euclidean của input với tất cả ảnh trong data
    distance_euclidean = np.array([])
    for i, dt_data in enumerate(dt_features_list):
        distance_euclidean = np.append(distance_euclidean, khoangcach_euclidean(dt_data, input_image_dt))
    distance_euclidean = chuan_hoa_euclidean_ve_khoang(distance_euclidean)
    return distance_euclidean

def knn(X_train, Y_train, X_new, k):
    hog = danhsach_euclidean(X_new[0], X_train[0])
    rgb = danhsach_euclidean(X_new[1], X_train[1])
    lbp = danhsach_euclidean(X_new[2], X_train[2])
    rgb1 = danhsach_euclidean(X_new[3], X_train[3])

    hog_rgb_similarities = 0.5*rgb1 + 0.3*hog + 0.2*lbp
    #hog_rgb_similarities = rgb1 + hog + 0.2*lbp
    
    sorted_similarities_euclidean = np.argsort(hog_rgb_similarities)[::1]

    # Chọn k điểm gần nhất
    top_similar_images = []
    for idx in sorted_similarities_euclidean[:k]:
        image_name = Y_train[0][idx]
        similarity = hog_rgb_similarities[idx]
        top_similar_images.append((image_name, similarity))
    return top_similar_images

#Đừờng dẫn ănh input
input_path = 'DATA/156.png'

#Đương dẫn của file lưu các bộ đặc trưng
path_hog ="DACTRUNG/hog.npy"
path_rgb ="DACTRUNG/rgb.npy"
path_lbp ="DACTRUNG/lbp.npy"
path_rgb1 ="DACTRUNG/rgball.npy"


data_hog = np.load(path_hog , allow_pickle="True")
data_rgb = np.load(path_rgb , allow_pickle="True")
data_lbp = np.load(path_lbp , allow_pickle="True")
data_rgb1 = np.load(path_rgb1 , allow_pickle="True")

#Sắp xếp các đặc trưng vào danh sách riêng
data_name_list = [item['image_name'] for item in data_hog] 
hog_features_list = [item['hog_features'] for item in data_hog] 
rgb_features_list = [item['rgb_features'] for item in data_rgb]
lbp_features_list = [item['lbp_features'] for item in data_lbp]
rgb1_features_list = [item['rgb_features'] for item in data_rgb1]

#Trích rút các đặc trưng của ảnh input
input_hog_features = convert_to_hog(input_path)
input_rgb_features = convert_to_rgb(input_path)
input_lbp_features = convert_to_lbp(input_path)
input_rgb1_features = convert_to_rgb1(input_path)

#Đưa đặc trưng vào tập train và test để chuẩn bị
X_new = [input_hog_features, input_rgb_features, input_lbp_features, input_rgb1_features]
X_train = [hog_features_list, rgb_features_list, lbp_features_list, rgb1_features_list]
Y_train = [data_name_list]

# Lấy ra danh sách ảnh giống nhất
k = 4
top_similar_images = knn(X_train, Y_train, X_new, k)

# In ra thông tin tat ca cac anh
for idx, (img_file, dissimilarity) in enumerate(top_similar_images):
    print(f"STT: {idx} - {img_file} - loss: {dissimilarity}")
print('\n')

input_image = cv2.imread(input_path)
input_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
plt.subplot(3, 2, 1)
plt.imshow(input_rgb)
plt.axis('off')
plt.title('Input Image')

# Vẽ ảnh khác biệt nhất từ kết hợp
for idx, (img_file, similarity_combined) in enumerate(top_similar_images):
    img_path = os.path.join(images_dir, img_file)
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
    plt.subplot(3, 2, idx+3)  # Vẽ ảnh trên 1 dòng, 3 cột
    plt.imshow(image_rgb)
    plt.axis('off')  # Tắt trục tọa độ
    plt.title(f'{img_file}\n loss: {similarity_combined:.9f}')  # Hiển thị độ tuong dong

plt.tight_layout()  # Đảm bảo layout hợp lý
plt.show()