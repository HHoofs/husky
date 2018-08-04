import os
import random

import numpy as np
from PIL import Image


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


def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


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
