# Python Modules
import os
import numpy as np
import cv2


def dilate_mask(pet, pet_cutoff=2, dilate_mm=15, brain_cutoff=85, assert_z_length=92):
    assert len(np.shape(pet)) == 3, f"needs to be a 3d image; current shape {np.shape(pet)}"
    assert np.shape(pet)[2] == assert_z_length, f"confirm that input is in z last; current z value {np.shape(pet)[2]}"

    # Create general mask that includes SUV values above pet_cutoff
    pet_mask = np.where(pet > pet_cutoff, 1, 0)

    # trying to removing the brain pet
    pet_mask[..., brain_cutoff:] = np.zeros(np.shape(pet_mask[..., brain_cutoff:]))
    pet_mask_dilate = np.zeros(np.shape(pet_mask))

    for i in range(np.shape(pet_mask)[2]):
        pet_mask_dilate[i, ...] = cv2.dilate(np.uint8(pet_mask[i, ...]),
                                             kernel=np.ones((dilate_mm, dilate_mm), 'uint8'),
                                             iterations=1)

    return pet_mask_dilate
