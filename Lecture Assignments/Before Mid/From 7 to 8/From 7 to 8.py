import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import median_filter

# Define your filter functions (same as before)
def average_filter(image, kernel_size=3):
    return cv2.blur(image, (kernel_size, kernel_size))

def circular_filter(image, radius=3):
    kernel = np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=np.float32)
    cy, cx = radius, radius
    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            if (y - cy)**2 + (x - cx)**2 <= radius**2:
                kernel[y, x] = 1
    kernel /= np.sum(kernel)
    return cv2.filter2D(image, -1, kernel)

def gaussian_filter(image, kernel_size=5, sigma=1.0):
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = kernel @ kernel.T
    return cv2.filter2D(image, -1, kernel)

def median_filter_function(image, kernel_size=3):
    return median_filter(image, size=kernel_size)

def kuwahara_filter(image, kernel_size=5):
    height, width = image.shape
    padding = kernel_size // 2
    output_image = np.zeros_like(image)
    for i in range(padding, height - padding):
        for j in range(padding, width - padding):
            regions = [
                image[i - padding:i + padding + 1, j - padding:j + padding + 1],
                image[i - padding:i + padding + 1, j:j + kernel_size],
                image[i:i + kernel_size, j - padding:j + padding + 1],
                image[i:i + kernel_size, j:j + kernel_size]
            ]
            region_means = [np.mean(region) for region in regions]
            region_variances = [np.var(region) for region in regions]
            min_var_index = np.argmin(region_variances)
            output_image[i, j] = np.mean(regions[min_var_index])
    return output_image

def laplacian_filter(image):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    return cv2.filter2D(image, -1, kernel)

def prewitt_filter(image):
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]])
    grad_x = cv2.filter2D(image, -1, kernel_x)
    grad_y = cv2.filter2D(image, -1, kernel_y)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return np.uint8(np.clip(grad_magnitude, 0, 255))

# Load the image
image = cv2.imread('Input Grayscale Photography Of Woman.jpg', cv2.IMREAD_GRAYSCALE)

# Apply filters
avg_filtered = average_filter(image)
circular_filtered = circular_filter(image)
gaussian_filtered = gaussian_filter(image)
median_filtered = median_filter_function(image)
kuwahara_filtered = kuwahara_filter(image)
laplacian_filtered = laplacian_filter(image)
prewitt_filtered = prewitt_filter(image)

# Display results using matplotlib
plt.figure(figsize=(12, 10))

# Original Image
plt.subplot(3, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')

# Apply and display each filter
plt.subplot(3, 3, 2), plt.imshow(avg_filtered, cmap='gray'), plt.title('Average Filter')
plt.subplot(3, 3, 3), plt.imshow(circular_filtered, cmap='gray'), plt.title('Circular Filter')
plt.subplot(3, 3, 4), plt.imshow(gaussian_filtered, cmap='gray'), plt.title('Gaussian Filter')
plt.subplot(3, 3, 5), plt.imshow(median_filtered, cmap='gray'), plt.title('Median Filter')
plt.subplot(3, 3, 6), plt.imshow(kuwahara_filtered, cmap='gray'), plt.title('Kuwahara Filter')
plt.subplot(3, 3, 7), plt.imshow(laplacian_filtered, cmap='gray'), plt.title('Laplacian Filter')
plt.subplot(3, 3, 8), plt.imshow(prewitt_filtered, cmap='gray'), plt.title('Prewitt Filter')

# Adjust layout and show
plt.tight_layout()
plt.show()
