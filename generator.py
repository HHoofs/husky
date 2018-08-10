import keras
import numpy as np
from skimage.transform import resize

from image_handling.augment import augment
from image_handling.utils import read_img_to_array, masks_as_image, random_crop_image_and_mask, \
    no_overlapping_boolean_arrays

from models.utils import select_pre_processing


class DataGenerator(keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, img_encodings, mask_type='c', path=None, batch_size=8, shuffle=True,
                 crop_dim=None, out_dim_img=(299, 299), out_dim_mask=None, classification=False,
                 encoder_model='VGG16'):
        # Initialization
        self.list_ids = list(img_encodings.keys())
        self.path = path
        self.img_encodings = img_encodings
        self.mask_type = mask_type
        self._output_channels_options(mask_type)
        self.indexes = np.arange(len(self.list_ids))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.crop_dim = crop_dim
        self.out_dim_img = out_dim_img
        self.out_dim_mask = self.out_dim_mask if out_dim_mask else out_dim_img
        self.add_classification = classification
        self.preprocessor_func = select_pre_processing(encoder_model)
        self.on_epoch_end()

    def _output_channels_options(self, mask_type):
        if mask_type.lower() == 'binary' or mask_type.lower() == 'b':
            self.output_channels = 1
            self.border_back_ground = False
        if mask_type.lower() == 'categorical' or mask_type.lower() == 'c':
            self.output_channels = 3
            self.border_back_ground = True

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
        _input = {'image': np.zeros((self.batch_size, *self.out_dim_img, 3))}
        _output = {'mask': np.zeros((self.batch_size, *self.out_dim_mask, self.output_channels))}
        if self.add_classification:
            _output['classification'] = np.zeros((self.batch_size, 1))

        for i, sample in enumerate(list_ids_temp):
            x_arr = read_img_to_array(sample)
            y_arr = masks_as_image(self.img_encodings[sample]['encoding'],
                                   self.border_back_ground,
                                   self.border_back_ground)
            if self.output_channels > 1:
                y_arr = [y_arr[2], y_arr[0], y_arr[1]]
                y_arr = no_overlapping_boolean_arrays(y_arr)
                y_arr = np.concatenate(y_arr, -1)
            x_arr, y_arr = self._if_needed_crop_resize(x_arr, y_arr)
            # augment
            x_arr, y_arr = augment((x_arr, y_arr))
            x_arr = self.preprocessor_func(x_arr)

            # store
            _input['image'][i] = x_arr
            _output['mask'][i] = y_arr
            # if classification - add to model
            if self.add_classification:
                _output['classification'][i] = np.max(y_arr)

        return _input, _output

    def _if_needed_crop_resize(self, x_arr, y_arr):
        _x_arr, _y_arr = x_arr, y_arr
        if self.crop_dim:
            if x_arr.shape[0] > self.crop_dim[0] and x_arr.shape[1] > self.crop_dim[1]:
                _x_arr, _y_arr = random_crop_image_and_mask(_x_arr, _y_arr, self.crop_dim)

        if _x_arr.shape != self.out_dim_img:
            _x_arr = resize(_x_arr, output_shape=(*self.out_dim_img, 3))

        if _y_arr.shape != self.out_dim_mask:
            _y_arr = resize(_y_arr, output_shape=(*self.out_dim_mask, self.output_channels))

        return _x_arr, _y_arr
