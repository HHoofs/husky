import keras
import numpy as np

from ski

from image_handling.augment import augment
from image_handling.utils import read_img_to_array, masks_as_image, random_crop_image_and_mask


class DataGenerator(keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, list_ids, path, img_encodings, batch_size=8, shuffle=True,
                 crop_dim=None, out_dim_img=(299,299), out_dim_mask=None, classification=False):
        # Initialization
        self.list_ids = list_ids
        self.path = path
        self.img_encodings =  img_encodings
        self.on_epoch_end()
        self.indexes = np.arange(len(self.list_ids))
        self.batch_size= batch_size
        self.shuffle = shuffle
        self.crop_dim = crop_dim
        self.out_dim_img = out_dim_img
        self.out_dim_mask = self.out_dim_mask if out_dim_mask else out_dim_img
        self.add_classifcation = classification

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        x, y = self.__data_generation(list_ids_temp)

        return x, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        input = {'images': np.zeros((self.batch_size, *self.out_dim_img, 3))}
        output = {'mask': np.zeros((self.batch_size, *self.out_dim_mask, 1))}
        if self.add_classifcation:
            output['classification'] = np.zeros((self.batch_size, 1))

        for i, sample in enumerate(list_ids_temp):
            x_arr = read_img_to_array(sample)
            y_arr = masks_as_image(self.img_encodings[sample]['encoding'])
            x_arr, y_arr = self._if_needed_crop(x_arr, y_arr)
            # augment
            x_arr, y_arr = augment(x_arr, y_arr)
            x_arr *= 255.0 / x_arr.max()

            # store
            input['images'][i] = x_arr
            output['mask'][i] = y_arr
            if self.add_classifcation:
                output['classification'][i] = np.max(y_arr)

        return input, output

    def _if_needed_crop(self, x_arr, y_arr):
        if self.crop_dim:
            if x_arr.shape[0] > self.crop_dim[0] and x_arr.shape[1] > self.crop_dim[1]:
                return random_crop_image_and_mask(x_arr, y_arr, self.crop_dim)

        return x_arr, y_arr