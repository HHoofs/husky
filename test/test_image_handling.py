import matplotlib.pyplot as plt
import numpy as np

from file_handling.read_db import read_csv_encoding, ids_for_each_count
from image_handling.add_ships import AddShips
from image_handling.augment import augment
from image_handling.utils import read_img_to_array, masks_as_image

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
    _plot_masks_borders(img, mask, borders)

    mask_without_border = masks_as_image(ID_MASK_ENCODING[img_id]['encoding'], add_borders_array=False)

    assert np.sum(mask) < np.sum(mask_without_border), 'boundaries should be deducted from mask'


def _plot_masks_borders(img, mask, borders):
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
    plt.imshow(borders[:, :, 0], cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('Added ships (mask)')
    plt.subplot(2, 2, 4)
    plt.imshow(np.add(mask, borders*2)[:, :, 0])
    plt.axis('off')
    plt.title('Added ships (mask)')
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
    plt.title('Added ships (mask)')
    plt.savefig('test/add_ships.png')
