import collections
import random
import numpy as np

from .utils import crop_mask_and_image, masks_as_image, read_img_to_array

class AddShips():
    def __init__(self, img_encoding, filled_ids):
        self.img_encoding = img_encoding
        self.filled_ids = list(collections._chain(*filled_ids.values()))

    def add_ship_to_img_and_mask(self, img, mask, n=1):
        ids = random.sample(self.filled_ids, n)
        new_image = img.copy()
        new_mask = mask.copy()
        for new_ship_id in ids:
            new_ship_mask = masks_as_image([random.choice(self.img_encoding[new_ship_id]['encoding'])])
            new_ship_img = read_img_to_array(new_ship_id)
            _, cropped_mask, ship_cut_out = crop_mask_and_image(mask=new_ship_mask, image=new_ship_img)
            dif_shape = [org - crop for org, crop in zip(img.shape, ship_cut_out.shape)]
            new_image, new_mask = self._join_anchor_and_new(dif_shape, new_image, new_mask, ship_cut_out, cropped_mask)

        return new_image, new_mask

    def _join_anchor_and_new(self, dif_shape, anchor_image, anchor_mask, new_image, new_mask):
        while True:
            x_left = random.randint(0, dif_shape[0] - new_image.shape[0])
            y_up = random.randint(0, dif_shape[1] - new_image.shape[1])

            place_in_mask = np.zeros_like(anchor_mask)
            place_in_mask[x_left:x_left + new_image.shape[0], y_up:y_up + new_image.shape[1]] = new_mask
            if not np.any(place_in_mask + anchor_mask == 2):
                updated_mask = anchor_mask + place_in_mask
                break

        place_in_img = np.zeros_like(anchor_image)
        place_in_img[x_left:x_left + new_image.shape[0], y_up:y_up + new_image.shape[1]] = new_image
        updated_img = np.where(place_in_mask == 1, place_in_img, anchor_image)

        return updated_img, updated_mask
