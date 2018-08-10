import os
import random

import numpy as np
from PIL import Image
from skimage.segmentation import find_boundaries


def read_img_to_array(sample):
    with Image.open(os.path.join('data', 'train', sample[:2], sample + '.jpg')) as _img_handler:
        return np.array(_img_handler)


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list, add_borders_array=False, add_background_array=False):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    all_borders = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            mask = rle_decode(mask)
            all_masks += mask
            if add_borders_array:
                all_borders = np.maximum(find_boundaries(mask, mode='thick').astype(int), all_borders)
    all_masks = all_masks.astype(bool)

    if not add_borders_array and not add_background_array:
        return np.expand_dims(all_masks, -1)

    _output = [np.expand_dims(all_masks, -1)]

    if add_borders_array:
        _output.append(np.expand_dims(all_borders.astype(bool), -1))

        if add_background_array:
            all_background = np.where(all_borders, False, True)
            all_background = np.where(all_masks, False, all_background)
            _output.append(np.expand_dims(all_background, -1))

    return _output


def no_overlapping_boolean_arrays(arrays):
    for anchor in range(len(arrays)):
        for reference in range(len(arrays)):
            if anchor < reference:
                arrays[anchor] = np.where(arrays[reference], False, arrays[anchor])

    return arrays


def crop_mask_and_image(mask, image):
    coords = np.argwhere(mask[:,:,0] == 1)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)

    res_mask = mask[x0:x1, y0:y1,:]
    res_img = image[x0:x1, y0:y1,:]
    res_img_cut_out = np.where(res_mask==1, res_img, 0)

    return res_img, res_mask, res_img_cut_out


def random_crop_image_and_mask(image, mask, size):
    x_start = random.randint(0, mask.shape[0] - size[0])
    y_start = random.randint(0, mask.shape[1] - size[1])

    return image[x_start:(x_start+size[0]), y_start:(y_start+size[1]), :], \
           mask[x_start:(x_start+size[0]), y_start:(y_start+size[1]), :]


def normalize(array):
    array *= 255.0 / array.max()
    return array
