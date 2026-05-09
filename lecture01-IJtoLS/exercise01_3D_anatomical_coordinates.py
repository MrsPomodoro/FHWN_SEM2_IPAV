#author: Barbara Klimek
# 3D case -> IJKtoLPS
# here we have 4x4 augmented matrix (IJKtoLPS)
# encode in the matrix a rotation submatrix (3x3), 3 component translation vector
# therefore, we have to determine 12 unknowns
# last row of the affine 3D transformation will be [0, 0, 0, 1]
#
#the vector of unknowns:
# A_1_1, A_1_2, A_1_3,
# A_2_1, A_2_2, A_2_3,
# A_3_1, A_3_2, A_3_3,
# t_x, t_y, t_z
#
# A_1_1 ... x-component of the first basis vector
# A_1_2 ... x-component of the second basis vector
# A_1_3 ... x-component of the third basis vector
# A_2_1 ... y-component of the first basis vector
# A_2_2 ... y-component of the second basis vector
# A_2_3 ... y-component of the third basis vector
# A_3_1 ... z-component of the first basis vector
# A_3_2 ... z-component of the second basis vector
# A_3_3 ... z-component of the third basis vector
#
# assume: origin: (50mm, 300mm, -25mm)
# and spacing: (50mm, 50mm, 50mm)
#
# system of linear equations to solve for unknowns:
# 12 equations to solve for the 12 unknowns
# if the origin of the coordinate system and the spacing are known,
# we can arrange the equation system as matrix S
#
# we need to solve the equation S * a = b
# b = left hand side of above system of linear equations
# a = thing that we want to know
# S will be set up according to the equations above
#
# the final solution are the entries of the matrix IJKtoLPS which
# transforms a homogeneous 3D image coordinate (4 component column vector)
# in homogeneous coordinates (i,j,k,1)
# to a 3D homogeneous coordinate (l,p,s,1) in the anatomical coordinate system

import numpy as np
# initialization of matrix
IJKtoLPS = np.zeros((4, 4))


# definition of  anatomical origin and voxel spacing in mm
origin = np.array([50, 300, -25])
spacing = np.array([50, 50, 50])

# helpers
def print_np_array(arr, caption_text):
    print(caption_text, '\n', arr, '\n', arr.shape, '\n')


# transformation matrix
IJKtoLPS = np.array([
    [spacing[0], 0,          0,          origin[0]],
    [0,          spacing[1], 0,          origin[1]],
    [0,          0,          spacing[2], origin[2]],
    [0,          0,          0,                  1]
])

print_np_array(IJKtoLPS, "IJKtoLPS:")

# transform voxel [0,0,0] to anatomical coordinates
pos1 = IJKtoLPS @ [0, 0, 0, 1]
print_np_array(pos1, "pos1:")

# transform voxel [1,1,1] to anatomical coordinates
pos2 = IJKtoLPS @ [1, 1, 1, 1]
print_np_array(pos2, "pos2:")

pos3 = IJKtoLPS @ [80, 120, 30, 1]
print_np_array(pos3, "pos3:")

#inverse transformation matrix - ransforming anatomical coordinates (LPS) back to image voxel coordinates (IJK)
LPStoIJK = np.linalg.inv(IJKtoLPS)
print_np_array(LPStoIJK, "LPStoIJK:")

# transform anatomical coordinates back to voxel coordinates
# this verifies that the transformation works correctly
pos3_in_ijk = LPStoIJK @ pos3
print_np_array(pos3_in_ijk, "pos3_in_ijk:")
