import cv2
import numpy as np
import matplotlib.pyplot as plt 

# 1. Read image
def read_image(filepath):
    image = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found at {filepath}")
    return image

# 2. Implement Histogram Equalization and Contrast Stretching

# 2.1. Histogram Equalization
def histogram_equalization(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(image)
    return equalized_image

# 2.2. Contrast Stretching
def contrast_stretching(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val, max_val = np.min(image), np.max(image)
    stretched_image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return stretched_image

# 3. Implement Filters: Averaging, Gaussian, and Median

# 3.1. Averaging Filter
def apply_averaging_filter(image, kernel_size=(5, 5)):
    return cv2.blur(image, kernel_size)

# 3.2. Gaussian Filter
def apply_gaussian_filter(image, kernel_size=(5, 5), sigma=1):
    return cv2.GaussianBlur(image, kernel_size, sigma)

# 3.3. Median Filter
def apply_median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

# Example of using the functions:

# Load the image
image_path = 'Grayscale Photography Of Woman.jpg'
image = read_image(image_path)

# 2.1. Histogram Equalization
equalized_image = histogram_equalization(image)

# 2.2. Contrast Stretching
stretched_image = contrast_stretching(image)


# 3.1.1 Apply Averaging Filter on orginal Image
averaged_original = apply_averaging_filter(image)

# 3.1.2 Apply Averaging Filter on Equalized Image
averaged_equalized = apply_averaging_filter(equalized_image)

# 3.1.3 Apply Averaging Filter on Stretched
averaged_stretched = apply_averaging_filter(stretched_image)


# 3.2.1 Apply Gaussian Filter on Original Image
gaussian_original = apply_gaussian_filter(image)

# 3.2.2 Apply Gaussian Filter on Equalized Image
gaussian_equalized = apply_gaussian_filter(equalized_image)

# 3.2.3 Apply Gaussian Filter on Stretched Image
gaussian_stretched = apply_gaussian_filter(stretched_image)


# 3.3.1 Apply Median Filter on Original Image
median_original = apply_median_filter(image)

# 3.3.2 Apply Median Filter on Equalized Image
median_equalized = apply_median_filter(equalized_image)

# 3.3.3 Apply Median Filter on Stretched Image
median_stretched = apply_median_filter(stretched_image)



# Display original and processed images
plt.figure(figsize=(15, 12))

# Original Images
plt.subplot(4, 3, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(4, 3, 2), plt.imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB)), plt.title('Equalized Image')
plt.subplot(4, 3, 3), plt.imshow(cv2.cvtColor(stretched_image, cv2.COLOR_BGR2RGB)), plt.title('Stretched Image')

# Averaging Filter Results
plt.subplot(4, 3, 4), plt.imshow(cv2.cvtColor(averaged_original, cv2.COLOR_BGR2RGB)), plt.title('Averaging - Original')
plt.subplot(4, 3, 5), plt.imshow(cv2.cvtColor(averaged_equalized, cv2.COLOR_BGR2RGB)), plt.title('Averaging - Equalized')
plt.subplot(4, 3, 6), plt.imshow(cv2.cvtColor(averaged_stretched, cv2.COLOR_BGR2RGB)), plt.title('Averaging - Stretched')

# Gaussian Filter Results
plt.subplot(4, 3, 7), plt.imshow(cv2.cvtColor(gaussian_original, cv2.COLOR_BGR2RGB)), plt.title('Gaussian - Original')
plt.subplot(4, 3, 8), plt.imshow(cv2.cvtColor(gaussian_equalized, cv2.COLOR_BGR2RGB)), plt.title('Gaussian - Equalized')
plt.subplot(4, 3, 9), plt.imshow(cv2.cvtColor(gaussian_stretched, cv2.COLOR_BGR2RGB)), plt.title('Gaussian - Stretched')

# Median Filter Results
plt.subplot(4, 3, 10), plt.imshow(cv2.cvtColor(median_original, cv2.COLOR_BGR2RGB)), plt.title('Median - Original')
plt.subplot(4, 3, 11), plt.imshow(cv2.cvtColor(median_equalized, cv2.COLOR_BGR2RGB)), plt.title('Median - Equalized')
plt.subplot(4, 3, 12), plt.imshow(cv2.cvtColor(median_stretched, cv2.COLOR_BGR2RGB)), plt.title('Median - Stretched')

plt.tight_layout()
plt.show()

