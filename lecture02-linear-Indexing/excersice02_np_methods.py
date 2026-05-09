# author: Barbara Klimek
#
# Linear indexing and basic image processing operations with NumPy
# Lecture 02 - Image Processing Algorithms and Visualization
#
# This example demonstrates:
# - image loading
# - image dimensions
# - pixel access
# - linear indexing
# - flattening arrays
# - reshaping arrays
# - image statistics
# - cropping
# - thresholding
# - histogram visualization
#
# The image used in this example is:
# moon.tif

import numpy as np
import cv2
import matplotlib.pyplot as plt

# load grayscale image without automatic conversion
I = cv2.imread('../lecture02-linear-Indexing/images/moon.tif', cv2.IMREAD_UNCHANGED)

# verify that image was loaded correctly
if I is None:
    raise FileNotFoundError("Image not found.")

# helper function for printing array information
def print_np_array(arr, caption_text):
    print(caption_text)
    print(arr)
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)
    print()


# ---------------------------------------------------------
# 1. IMAGE DIMENSIONS
# ---------------------------------------------------------
# shape returns image dimensions
# for grayscale image -> (height, width)

print_np_array(I, "Original image:")

height, width = I.shape

print("height:", height)
print("width:", width)
print()


# ---------------------------------------------------------
# 2. PIXEL ACCESS USING 2D INDEXING
# ---------------------------------------------------------
# access pixel intensity at position [y,x]

x = 132
y = 223

pixel_value = I[y, x]

print("pixel intensity at [y,x]:", pixel_value)
print()


# ---------------------------------------------------------
# 3. LINEAR INDEX COMPUTATION
# ---------------------------------------------------------
# convert 2D image coordinates into 1D linear index

linear_index = width * y + x

print("linear index:", linear_index)
print()


# ---------------------------------------------------------
# 4. FLATTEN IMAGE INTO 1D ARRAY
# ---------------------------------------------------------
# flatten converts 2D image into one long vector

flat_image = I.flatten()

print_np_array(flat_image, "Flattened image:")


# ---------------------------------------------------------
# 5. ACCESS PIXEL USING LINEAR INDEX
# ---------------------------------------------------------
# verify that linear indexing returns the same pixel value

pixel_linear = flat_image[linear_index]

print("pixel value from linear indexing:", pixel_linear)
print()


# ---------------------------------------------------------
# 6. RESHAPE ARRAY BACK TO ORIGINAL IMAGE
# ---------------------------------------------------------
# reshape restores original image dimensions

restored_image = flat_image.reshape(I.shape)

print_np_array(restored_image, "Restored image:")


# ---------------------------------------------------------
# 7. IMAGE STATISTICS
# ---------------------------------------------------------
# compute minimum, maximum and mean intensity

print("minimum intensity:", np.min(I))
print("maximum intensity:", np.max(I))
print("mean intensity:", np.mean(I))
print()


# ---------------------------------------------------------
# 8. IMAGE CROPPING
# ---------------------------------------------------------
# extract small image region

crop = I[100:250, 100:250]

print_np_array(crop, "Cropped image region:")

plt.figure()
plt.imshow(crop, cmap='gray')
plt.title("Cropped Region")


# ---------------------------------------------------------
# 9. IMAGE THRESHOLDING
# ---------------------------------------------------------
# convert grayscale image into binary image

binary_image = I > 120

print_np_array(binary_image, "Binary image:")

plt.figure()
plt.imshow(binary_image, cmap='gray')
plt.title("Thresholded Image")


# ---------------------------------------------------------
# 10. IMAGE HISTOGRAM
# ---------------------------------------------------------
# histogram shows intensity distribution

plt.figure()
plt.hist(I.flatten(), bins=256)
plt.title("Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")


# ---------------------------------------------------------
# SHOW ORIGINAL IMAGE
# ---------------------------------------------------------

plt.figure()
plt.imshow(I, cmap='gray')
plt.title("Original Image")

plt.show()