import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

from file_handling.read_db import read_csv_encoding, ids_for_each_count
from image_handling.add_ships import AddShips
from image_handling.augment import augment
from image_handling.utils import read_img_to_array, masks_as_image, no_overlapping_boolean_arrays

ID_MASK_ENCODING = read_csv_encoding(length=50)


def test_add_ships(plot_test=True):
    img_id = '00113a75c'
    img = read_img_to_array(img_id)
    empty_ids, filled_ids = ids_for_each_count(ID_MASK_ENCODING)
    mask = masks_as_image(ID_MASK_ENCODING[img_id]['encoding'])

    ship_adder = AddShips(img_encoding=ID_MASK_ENCODING, filled_ids=filled_ids)
    new_image, new_mask = ship_adder.add_ship_to_img_and_mask(img=img, mask=mask, n=10)

    if plot_test:
        _plot_masks_images(img, mask, new_image, new_mask)

    assert np.sum(mask) < np.sum(new_mask), print('nothing added in the mask')


def test_augment():
    array = np.array([[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]]).reshape((4, 4, 1))
    equal_arrays = (array, array)
    array_1, array_2 = augment(equal_arrays)

    assert np.array_equal(array_1, array_2), 'augmentation resulted arrays that are not equal'


def test_mask_as_image():
    img_id = '00113a75c'
    img = read_img_to_array(img_id)
    mask, borders = masks_as_image(ID_MASK_ENCODING[img_id]['encoding'], add_borders_array=True)
    _mask, _borders = no_overlapping_boolean_arrays([mask, borders])
    mask_without_border = masks_as_image(ID_MASK_ENCODING[img_id]['encoding'])

    _plot_masks_borders(img, mask, borders)

    assert np.sum(borders) == np.sum(_borders), 'borders should be diminished'
    assert np.sum(_mask) < np.sum(mask_without_border), 'boundaries should be deducted from mask'


def test_mask_border_background_overlap():
    img_id = '00113a75c'
    mask, borders, background = masks_as_image(ID_MASK_ENCODING[img_id]['encoding'],
                                               add_borders_array=True,
                                               add_background_array=True)
    mask, borders, background = no_overlapping_boolean_arrays([mask, borders, background])

    number_of_trues = np.sum(mask) + np.sum(borders) + np.sum(background)

    assert number_of_trues == mask.shape[1] * mask.shape[0], 'each cell should have a True'


def test_augment_and_resize():
    sample = '000155de5'
    x_arr = read_img_to_array(sample)
    y_arr = masks_as_image(ID_MASK_ENCODING[sample]['encoding'])
    x_arr, y_arr = _if_needed_crop_resize(x_arr, y_arr, (224, 224), (224, 224))
    # augment
    x_arr, y_arr = augment((x_arr, y_arr))

    _plot_resized_mask(x_arr, y_arr)

    assert np.max(y_arr) == 1, 'max value of resized mask should be 1'
    assert np.min(y_arr) == 0, 'min value of resized mask should be 0'


def _plot_masks_borders(img, mask, borders):
    plt.figure(figsize=(6.45, 6.45), dpi=300)
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Image')
    plt.subplot(2, 2, 2)
    plt.imshow(mask[:, :, 0], cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('Mask')
    plt.subplot(2, 2, 3)
    plt.imshow(borders[:, :, 0], cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('Border')
    plt.subplot(2, 2, 4)
    plt.imshow(np.add(mask, borders*2)[:, :, 0])
    plt.axis('off')
    plt.title('Mask and border')
    plt.savefig('test/mask_borders.png')


def _plot_masks_images(img, mask, new_image, new_mask):
    plt.figure(figsize=(6.45, 6.45), dpi=300)
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original ships (image)')
    plt.subplot(2, 2, 2)
    plt.imshow(mask[:, :, 0], cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('Original ships (mask)')
    plt.subplot(2, 2, 3)
    plt.imshow(new_image)
    plt.axis('off')
    plt.title('Added ships (image)')
    plt.subplot(2, 2, 4)
    plt.imshow(new_mask[:, :, 0], cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('Border and mask')
    plt.savefig('test/add_ships.png')


def _plot_resized_mask(x_arr, y_arr):
    plt.subplot(1, 2, 1)
    plt.imshow(x_arr)
    plt.axis('off')
    plt.title('Image')
    plt.subplot(1, 2, 2)
    plt.imshow(y_arr[:, :, 0], cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('Resized mask')
    plt.savefig('test/resized_image.png')


def _if_needed_crop_resize(x_arr, y_arr, out_dim_img, out_dim_mask):
    _x_arr, _y_arr = x_arr, y_arr

    if _x_arr.shape != out_dim_img:
        _x_arr = resize(_x_arr, output_shape=out_dim_img)

    if _y_arr.shape != out_dim_mask:
        _y_arr = resize(_y_arr, output_shape=out_dim_mask)

    return _x_arr, _y_arr
