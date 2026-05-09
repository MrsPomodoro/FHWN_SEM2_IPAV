import numpy as np
import cv2
import matplotlib.pyplot as plt

# load image
I = cv2.imread('../lecture02-linear-Indexing/images/moon.tif', cv2.IMREAD_UNCHANGED)

print('image dimensions:', I.shape, '\n')

# image dimensions
dims = I.shape

# specify 2D index/sampling position
x = 132
y = 223

# pixel value using normal 2D indexing
intensity_by_2D_array_indexing = I[y, x]

print('intensity value at pos [y:', y, ', x:', x, ']:',
      intensity_by_2D_array_indexing)

# compute linear index manually
linear_index = dims[1] * y + x

print('linear index of the pixel position:', linear_index)

# pixel intensity using linear indexing
intensity_by_linear_index = I.flatten()[linear_index]

print('intensity value at position accessed via linear index:',
      intensity_by_linear_index)

# compute linear index using numpy
linear_index_by_numpy = np.ravel_multi_index(
    [y, x],
    dims,
    mode='raise',
    order='C'
)

print("linear index of the pixel position by numpy's ravel_multi_index:",
      linear_index_by_numpy)

# pixel intensity using numpy linear index
intensity_by_linear_index2 = I.flatten()[linear_index_by_numpy]

print('intensity value at position accessed via linear index:',
      intensity_by_linear_index2)

# show image
plt.figure()
plt.imshow(I, cmap='gray')
plt.show()