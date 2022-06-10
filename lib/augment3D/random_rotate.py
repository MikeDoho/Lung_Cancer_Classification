import numpy as np
import scipy.ndimage as ndimage


# def random_rotate3D(img_numpy, label, min_angle, max_angle):
#     """
#     Returns a random rotated array in the same shape
#     :param img_numpy: 3D numpy array
#     :param min_angle: in degrees
#     :param max_angle: in degrees
#     :return: 3D rotated img
#     """
#
#     assert img_numpy.ndim == 3, print("expected dim=3 but got " + str(img_numpy.shape))
#     assert min_angle < max_angle, "min should be less than max val"
#     assert min_angle > -360 or max_angle < 360
#     all_axes = [(1, 2)]
#     # all_axes = [(1, 0), (1, 2), (0, 2)]
#     angle = np.random.randint(low=min_angle, high=max_angle + 1)
#     axes_random_id = np.random.randint(low=0, high=len(all_axes))
#     axes = all_axes[axes_random_id]
#
#     dim1, dim2, dim3 = img_numpy.shape
#     temp_img = ndimage.rotate(img_numpy, angle, axes=axes)
#     temp_label = ndimage.rotate(label, angle, axes=axes)
#
#     p_dim1, p_dim2, p_dim3 = temp_img.shape
#
#     return temp_img[:, p_dim2//2-dim2//2:p_dim2//2-dim2//2+dim2, p_dim3//2-dim3//2:p_dim3//2-dim3//2+dim3], \
#            temp_label[:, p_dim2//2-dim2//2:p_dim2//2-dim2//2+dim2, p_dim3//2-dim3//2:p_dim3//2-dim3//2+dim3]

def random_rotate3D(img_numpy, label, min_angle, max_angle):
    """
    Returns a random rotated array in the same shape
    :param img_numpy: 3D numpy array
    :param min_angle: in degrees
    :param max_angle: in degrees
    :return: 3D rotated img
    """

    assert img_numpy.ndim == 3, print("expected dim=3 but got " + str(img_numpy.shape))
    assert min_angle < max_angle, "min should be less than max val"
    assert min_angle > -360 or max_angle < 360
    all_axes = [(1, 2)]
    # all_axes = [(1, 0), (1, 2), (0, 2)]
    angle = np.random.randint(low=min_angle, high=max_angle + 1)
    axes_random_id = np.random.randint(low=0, high=len(all_axes))
    axes = all_axes[axes_random_id]

    dim1, dim2, dim3 = img_numpy.shape
    temp_img = ndimage.rotate(img_numpy, angle, axes=axes)
    # temp_label = ndimage.rotate(label, angle, axes=axes)
    temp_label = label

    p_dim1, p_dim2, p_dim3 = temp_img.shape

    return temp_img[:, p_dim2 // 2 - dim2 // 2:p_dim2 // 2 - dim2 // 2 + dim2,
           p_dim3 // 2 - dim3 // 2:p_dim3 // 2 - dim3 // 2 + dim3], \
           temp_label



class RandomRotation(object):
    def __init__(self, min_angle=-10, max_angle=10):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, img_numpy, label=None):
        """
        Args:
            img_numpy (numpy): Image to be rotated.
            label (numpy): Label segmentation map to be rotated

        Returns:
            img_numpy (numpy): rotated img.
            label (numpy): rotated Label segmentation map.
        """
        # print(f"Rotation (min/max): {self.min_angle} / {self.max_angle}")

        img_numpy, label = random_rotate3D(img_numpy, label, self.min_angle, self.max_angle)
        return img_numpy, label
