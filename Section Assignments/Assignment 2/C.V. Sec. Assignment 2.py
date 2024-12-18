import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read grayscale image
def read_image(filepath):
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {filepath}")
    return image

# 2. Implement Derivative Filters: Laplacian, Sobel, and Prewitt Filters

# 2.1. Laplacian Filter
def apply_laplacian(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    return laplacian

# 2.2. Sobel Filter
def apply_sobel(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.sqrt(sobel_x**2 + sobel_y**2)
    sobel = cv2.convertScaleAbs(sobel)
    return sobel

# 2.3. Prewitt Filter
def apply_prewitt(image):
    prewitt_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    prewitt_y = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    
    prewitt_x_filtered = cv2.filter2D(image, -1, prewitt_x).astype(np.float32)
    prewitt_y_filtered = cv2.filter2D(image, -1, prewitt_y).astype(np.float32)
    prewitt = cv2.sqrt(prewitt_x_filtered**2 + prewitt_y_filtered**2)
    prewitt = cv2.convertScaleAbs(prewitt)
    return prewitt

# Example of using the functions:

# Load the image
image_path = 'Assignment 2\Input Grayscale Photography Of Woman.jpg'  
image = read_image(image_path)

# Apply filters
laplacian_image = apply_laplacian(image)
sobel_image = apply_sobel(image)
prewitt_image = apply_prewitt(image)

# Display original and edge-detected images
plt.figure(figsize=(12, 12))

# Original Image
plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')

# Laplacian Filter Results
plt.subplot(2, 2, 2), plt.imshow(laplacian_image, cmap='gray'), plt.title('Laplacian Edge Detection')

# Sobel Filter Results
plt.subplot(2, 2, 3), plt.imshow(sobel_image, cmap='gray'), plt.title('Sobel Edge Detection')

# Prewitt Filter Results
plt.subplot(2, 2, 4), plt.imshow(prewitt_image, cmap='gray'), plt.title('Prewitt Edge Detection')

plt.tight_layout()
plt.show()