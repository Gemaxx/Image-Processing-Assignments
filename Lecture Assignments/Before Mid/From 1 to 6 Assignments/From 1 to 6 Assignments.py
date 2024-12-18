import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Labeling of Connected Regions
def label_connected_regions(image):
    # Convert the image to binary
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # Find the connected components
    num_labels, labels = cv2.connectedComponents(binary_image)
    return num_labels, labels

# 2. Image Negative
def image_negative(image):
    negative_image = 255 - image
    return negative_image

# 3. Bit Plane Slicing
def bit_plane_slicing(image, plane):
    # Extract the specified bit plane (0-7)
    bit_plane = (image >> plane) & 1
    bit_plane = bit_plane * 255
    return bit_plane.astype(np.uint8)

# 4. Contrast Stretching and Contrast Contraction
def contrast_stretching(image):
    min_val, max_val = np.min(image), np.max(image)
    stretched_image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return stretched_image

def contrast_contraction(image):
    min_val, max_val = np.min(image), np.max(image)
    contracted_image = ((image - min_val) / (max_val - min_val) * 255 * 0.5).astype(np.uint8)
    return contracted_image

# 5. Gray Level Slicing
def gray_level_slicing(image, low, high):
    sliced_image = np.copy(image)
    sliced_image[(image >= low) & (image <= high)] = 255
    sliced_image[~((image >= low) & (image <= high))] = 0
    return sliced_image

# 6. Histogram Equalization
def histogram_equalization(image):
    equalized_image = cv2.equalizeHist(image)
    return equalized_image

# Example usage of the functions
image_path = 'Section Assignments/Assignment 2/Input Grayscale Photography Of Woman.jpg'  
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 1. Labeling of Connected Regions
num_labels, labeled_image = label_connected_regions(image)

# 2. Image Negative
negative_image = image_negative(image)

# 3. Bit Plane Slicing (Example for 3rd bit plane)
bit_plane_image = bit_plane_slicing(image, 3)

# 4. Contrast Stretching and Contraction
stretched_image = contrast_stretching(image)
contracted_image = contrast_contraction(image)

# 5. Gray Level Slicing (Example for slicing between 100 and 150)
sliced_image = gray_level_slicing(image, 100, 150)

# 6. Histogram Equalization
equalized_image = histogram_equalization(image)

# Display results
plt.figure(figsize=(12, 10))

# Original Image
plt.subplot(3, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')

# Labeling of Connected Regions
plt.subplot(3, 3, 2), plt.imshow(labeled_image, cmap='jet'), plt.title(f'Connected Components ({num_labels} Labels)')

# Image Negative
plt.subplot(3, 3, 3), plt.imshow(negative_image, cmap='gray'), plt.title('Image Negative')

# Bit Plane Slicing
plt.subplot(3, 3, 4), plt.imshow(bit_plane_image, cmap='gray'), plt.title('Bit Plane Slicing (3rd Bit)')

# Contrast Stretching
plt.subplot(3, 3, 5), plt.imshow(stretched_image, cmap='gray'), plt.title('Contrast Stretching')

# Contrast Contraction
plt.subplot(3, 3, 6), plt.imshow(contracted_image, cmap='gray'), plt.title('Contrast Contraction')

# Gray Level Slicing
plt.subplot(3, 3, 7), plt.imshow(sliced_image, cmap='gray'), plt.title('Gray Level Slicing (100-150)')

# Histogram Equalization
plt.subplot(3, 3, 8), plt.imshow(equalized_image, cmap='gray'), plt.title('Histogram Equalization')

plt.tight_layout()
plt.show()
