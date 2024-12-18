# CV Section Assignment 2 Documentation

## Overview

This assignment focuses on edge detection techniques in image processing using derivative filters. The task includes:

1. Reading a grayscale image for edge detection.
2. Implementing derivative filters: Laplacian, Sobel, and Prewitt.
3. Visualizing the results by displaying the original image and the edge-detected images using each filter side by side.
4. Discussing the strengths and weaknesses of each filter in terms of edge detection.

## Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib

## Functions Description

### 1. Read Image

Reads a grayscale image from the specified file path for processing.

```python
def read_image(filepath):
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {filepath}")
    return image
```

### 2. Derivative Filters

#### 2.1. Laplacian Filter

The Laplacian filter detects edges by computing the second derivative of the image.

```python
def apply_laplacian(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    return laplacian
```

#### 2.2. Sobel Filter

The Sobel filter detects edges by computing the first derivative in both the x and y directions.

```python
def apply_sobel(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.sqrt(sobel_x**2 + sobel_y**2)
    sobel = cv2.convertScaleAbs(sobel)
    return sobel
```

#### 2.3. Prewitt Filter

The Prewitt filter detects edges by applying convolution masks in the x and y directions.

```python
def apply_prewitt(image):
    prewitt_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    prewitt_y = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    
    prewitt_x_filtered = cv2.filter2D(image, -1, prewitt_x).astype(np.float32)
    prewitt_y_filtered = cv2.filter2D(image, -1, prewitt_y).astype(np.float32)
    prewitt = cv2.sqrt(prewitt_x_filtered**2 + prewitt_y_filtered**2)
    prewitt = cv2.convertScaleAbs(prewitt)
    return prewitt
```

## Steps to Execute

1. Place the input grayscale image (`Grayscale Photography Of Woman.jpg`) in the same directory as the script.
2. Run the script `Assignment 2: Image Edge Detection.py`.
3. The output will display the processed edge-detected images using Laplacian, Sobel, and Prewitt filters side by side.

## Example Visualization

### Displayed Images

- **Original Image**
- **Laplacian Edge Detection**
- **Sobel Edge Detection**
- **Prewitt Edge Detection**

### Sample Visualization Code

```python
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
```
### Input Example
![Original Image](<Input Grayscale Photography Of Woman.jpg>)

### Output Example
![Eadge Detection](<Output Comparison.png>)

## Comparison of Filters

### 1. Laplacian Filter
- **Strengths**: 
  - Detects edges by highlighting regions of rapid intensity change.
  - Works well for images with sharp edges.
- **Weaknesses**: 
  - Sensitive to noise, which may produce false edges.
  - Can be less effective in detecting subtle edges.

### 2. Sobel Filter
- **Strengths**: 
  - Good for detecting edges in horizontal and vertical directions.
  - Can emphasize edge structures in both x and y directions.
- **Weaknesses**: 
  - Less accurate in detecting edges at other angles.
  - May result in blurred edges due to averaging.

### 3. Prewitt Filter
- **Strengths**: 
  - Similar to Sobel, but with a slightly different convolution mask.
  - Effective for detecting edges in both horizontal and vertical directions.
- **Weaknesses**: 
  - Also prone to blurring and less sensitive to diagonal edges.
  - More sensitive to noise compared to Sobel.

#### By Ahmed Ibrahim Metwally Negm - ID: 322223887
#### Supervisors: Dr. Shimaa Othman, Eng. Yosr
Thank you, ✨🤎.