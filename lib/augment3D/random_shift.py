import numpy as np
import scipy.ndimage as ndimage


# def transform_matrix_offset_center_3d(matrix, x, y, z):
#     print(np.shape(matrix))
#     dim1, dim2, dim3 = np.shape(matrix)
#     print(dim1, dim2, dim3)
#     # dim1, dim2, dim3 = matrix.shape
#     offset_matrix = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
#     # return ndimage.interpolation.affine_transform(matrix, offset_matrix)
#     temp_img = ndimage.interpolation.affine_transform(matrix, offset_matrix)
#
#     p_dim1, p_dim2, p_dim3 = temp_img.shape
#
#     return temp_img[:, p_dim2 // 2 - dim2 // 2:p_dim2 // 2 - dim2 // 2 + dim2,
#            p_dim3 // 2 - dim3 // 2:p_dim3 // 2 - dim3 // 2 + dim3]

def transform_matrix_offset_center_3d(matrix, x, y, z):
    # print(np.shape(matrix))
    dim1, dim2, dim3 = np.shape(matrix)
    # print(dim1, dim2, dim3)
    # dim1, dim2, dim3 = matrix.shape
    offset_matrix = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    # return ndimage.interpolation.affine_transform(matrix, offset_matrix)
    temp_img = ndimage.interpolation.affine_transform(matrix, offset_matrix)

    p_dim1, p_dim2, p_dim3 = temp_img.shape

    return temp_img[:, p_dim2 // 2 - dim2 // 2:p_dim2 // 2 - dim2 // 2 + dim2,
           p_dim3 // 2 - dim3 // 2:p_dim3 // 2 - dim3 // 2 + dim3]


# def random_shift(img_numpy, label, max_percentage=0.2):
#     # print("img_numpy.shape in random_shift:")
#     # print(img_numpy.shape)
#     dim1, dim2, dim3 = img_numpy.shape
#     m1, m2, m3 = int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2)
#     # print(m1, m2, m3)
#     # d1 = np.random.randint(-m1, m1)
#     d1 = 0
#     if m2 == 0:
#         d2 = 0
#     else:
#         d2 = np.random.randint(-m2, m2)
#     if m3 == 0:
#         d3 = 0
#     else:
#         d3 = np.random.randint(-m3, m3)
#     return transform_matrix_offset_center_3d(img_numpy, d1, d2, d3), transform_matrix_offset_center_3d(label, d1, d2,
#                                                                                                        d3)

def random_shift(img_numpy, label, max_percentage=0.2):
    # print("img_numpy.shape in random_shift:")
    # print(np.shape(img_numpy))
    dim1, dim2, dim3 = img_numpy.shape
    m1, m2, m3 = int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2)
    # print(m1, m2, m3)
    # d1 = np.random.randint(-m1, m1)
    d1 = 0
    if m2 == 0:
        d2 = 0
    else:
        d2 = np.random.randint(-m2, m2)
    if m3 == 0:
        d3 = 0
    else:
        d3 = np.random.randint(-m3, m3)
    return transform_matrix_offset_center_3d(img_numpy, d1, d2, d3), label


class RandomShift(object):
    def __init__(self, max_percentage=0.25):
        self.max_percentage = max_percentage

    def __call__(self, img_numpy, label=None):
        img_numpy, label = random_shift(img_numpy, label, self.max_percentage)
        return img_numpy, label


class RandomShift_tta(object):
    def __init__(self, max_percentage=0.30):
        self.max_percentage = max_percentage

    def __call__(self, img_numpy, label=None):
        # print(f"random shift: {self.max_percentage}")
        img_numpy, label = random_shift(img_numpy, label, self.max_percentage)
        return img_numpy, label
