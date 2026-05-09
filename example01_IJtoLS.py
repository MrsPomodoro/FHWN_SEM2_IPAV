# "easy" example in 2D case instead of IJKtoLPS (3D case)
# instead of 3D transformation with matrix A being 4x4 (augmented)
# matrix (IJKtoLPS)
# here we have matrix A being a 3x3 (augmented) matrix (IJtoLS)
# encode in the matrix a rotation submatrix (2x2), 2 component
# translation vector
# therefore, we have to determine 6 unknowns

import numpy as np

def print_np_array(arr, caption_text):
    print(caption_text, '\n', arr, '\n', arr.shape, '\n')

S = np.array([
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 1]
])

print_np_array(S, 'S (system matrix):')

b = np.array([50, 300, 100, 300, 50, 250])
print_np_array(b, 'b:')

# solve S * a = b
a = np.linalg.inv(S) @ b

print_np_array(a, 'a:')

IJtoLS = np.zeros((3, 3))

IJtoLS[0, 0] = a[0]
IJtoLS[0, 1] = a[1]
IJtoLS[1, 0] = a[2]
IJtoLS[1, 1] = a[3]
IJtoLS[0, 2] = a[4]
IJtoLS[1, 2] = a[5]
IJtoLS[2, 2] = 1

print_np_array(IJtoLS, 'IJtoLS:')

# tests
pos1 = IJtoLS @ [0, 0, 1]
print_np_array(pos1, 'pos1:')

pos2 = IJtoLS @ [1, 1, 1]
print_np_array(pos2, 'pos2:')

pos3 = IJtoLS @ [-1, -1, 1]
print_np_array(pos3, 'pos3:')

pos4 = IJtoLS @ [97, 137, 1]
print_np_array(pos4, 'pos4:')

# inverse transformation
LStoIJ = np.linalg.inv(IJtoLS)
print_np_array(LStoIJ, 'LStoIJ:')

pos4_in_ij = LStoIJ @ pos4
print_np_array(pos4_in_ij, 'pos4_in_ij:')