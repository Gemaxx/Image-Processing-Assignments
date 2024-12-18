import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Otsu's Thresholding
def otsu_thresholding(image):
    # Apply Otsu's Thresholding method
    _, otsu_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_image

# 2. Adaptive Thresholding
def adaptive_thresholding(image):
    # Apply adaptive thresholding using Gaussian mean
    adaptive_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
    return adaptive_image

# Example usage of the functions
image_path = 'Before Mid/From 7 to 8/Input Grayscale Photography Of Woman.jpg'  
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 1. Otsu's Thresholding
otsu_image = otsu_thresholding(image)

# 2. Adaptive Thresholding
adaptive_image = adaptive_thresholding(image)

# Display results
plt.figure(figsize=(12, 10))

# Original Image
plt.subplot(2, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')

# Otsu's Thresholding
plt.subplot(2, 3, 2), plt.imshow(otsu_image, cmap='gray'), plt.title("Otsu's Thresholding")

# Adaptive Thresholding
plt.subplot(2, 3, 3), plt.imshow(adaptive_image, cmap='gray'), plt.title('Adaptive Thresholding')

plt.tight_layout()
plt.show()
