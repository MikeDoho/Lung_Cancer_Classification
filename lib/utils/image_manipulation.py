#Modules
import numpy as np
import math


'''

Functions

'''
def pad_to_shape(x, shape=(48, 48, 48)):

    '''
    :param x: 3D image
    :param shape: output shape request
    :return: 3D image zero padded to shape
    '''

    assert len(np.shape(x)) == 3, 'current code requires input to be 3d'
    assert len(shape) == 3, 'current code requires 3d shape'

    for i, dim in enumerate(shape):
        if dim >= np.shape(x)[i]:
            pass
        else:
            raise ValueError(f"a dimension of the array is larger than the proposed shape: {np.shape(x)} vs {shape}")

    # dimension 0
    dim0_diff = shape[0] - np.shape(x)[0]
    dim0_diff_f = math.floor(dim0_diff / 2)
    dim0_diff_c = math.ceil(dim0_diff / 2)

    # dimension 1
    dim1_diff = shape[1] - np.shape(x)[1]
    dim1_diff_f = math.floor(dim1_diff / 2)
    dim1_diff_c = math.ceil(dim1_diff / 2)

    # dimension 2
    dim2_diff = shape[2] - np.shape(x)[2]
    dim2_diff_f = math.floor(dim2_diff / 2)
    dim2_diff_c = math.ceil(dim2_diff / 2)

    x = np.pad(x, ((dim0_diff_f, dim0_diff_c), (dim1_diff_f, dim1_diff_c), (dim2_diff_f, dim2_diff_c)))

    assert np.shape(x) == shape, f"shapes are not equal: {np.shape(x)} vs {shape}"
    return x

def crop_center(img, cropy, cropx, cropz):
    y, x, z = np.shape(img)
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    startz = z // 2 - (cropz // 2)
    return img[starty:starty + cropy, startx:startx + cropx, startz:startz + cropz]