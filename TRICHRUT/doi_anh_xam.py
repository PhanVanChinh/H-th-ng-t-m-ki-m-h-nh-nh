import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_grayscale(image):
    # Kiểm tra nếu ảnh là ảnh màu (3 kênh)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Trích xuất các kênh màu
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        # Áp dụng công thức chuyển đổi
        gray_image = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # Đảm bảo giá trị pixel trong khoảng từ 0 đến 255
        gray_image = gray_image.astype(np.uint8)
    else:
        # Nếu ảnh đã là ảnh xám thì trả về chính nó
        gray_image = image
    return gray_image

# Đường dẫn tới ảnh
image_path = 'TEST/anh160.png'

# Đọc ảnh
image = cv2.imread(image_path)

# Kiểm tra nếu ảnh được đọc thành công
if image is not None:
    # Chuyển đổi ảnh sang ảnh xám
    gray_image = rgb_to_grayscale(image)
    
    # Hiển thị ảnh gốc và ảnh xám sử dụng matplotlib
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Gray Image')
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')

    plt.show()
    
    # Lưu ảnh xám
    cv2.imwrite('gray_image.png', gray_image)
else:
    print(f"Could not read the image at {image_path}")
