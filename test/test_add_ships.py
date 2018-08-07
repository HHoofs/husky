import numpy as np
import matplotlib.pyplot as plt

from file_handling.read_db import read_csv_encoding, ids_for_each_count
from image_handling.add_ships import AddShips
from image_handling.utils import read_img_to_array, masks_as_image


def test_add_ships(plot_test=True):
    img_id = '00113a75c'
    img = read_img_to_array(img_id)
    id_masks_encoding = read_csv_encoding()
    empty_ids, filled_ids = ids_for_each_count(id_masks_encoding)
    mask = masks_as_image(id_masks_encoding[img_id]['encoding'])

    ship_adder = AddShips(img_encoding=id_masks_encoding, filled_ids=filled_ids)
    new_image, new_mask = ship_adder.add_ship_to_img_and_mask(img=img, mask=mask, n=10)

    if plot_test:
        _plot_masks_images(img, mask, new_image, new_mask)

    assert np.sum(mask) < np.sum(new_mask), print('nothing added in the mask')


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
