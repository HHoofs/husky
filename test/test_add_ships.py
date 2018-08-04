import numpy as np

from file_handling.read_db import read_csv_encoding, ids_for_each_count
from image_handling.add_ships import AddShips
from image_handling.utils import read_img_to_array, masks_as_image


def test_add_ships():
    img_id = '00113a75c'
    img = read_img_to_array(img_id)
    id_masks_encoding = read_csv_encoding()
    empty_ids, filled_ids = ids_for_each_count(id_masks_encoding)
    mask = masks_as_image(id_masks_encoding[img_id]['encoding'])

    ship_adder = AddShips(img_encoding=id_masks_encoding, filled_ids=filled_ids)
    new_image, new_mask = ship_adder.add_ship_to_img_and_mask(img=img, mask=mask, n=1)

    assert np.sum(mask) < np.sum(new_mask), print('nothing added in the mask')
