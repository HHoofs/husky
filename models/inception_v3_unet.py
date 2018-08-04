import keras
import keras.backend as K
import numpy as np
from keras import models, Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Dense, concatenate, MaxPooling2D, \
    Concatenate, AveragePooling2D, Cropping2D, Flatten, Input

from .models import PretrainedDecoderRawEncoderUnet


class InceptionV3Unet(PretrainedDecoderRawEncoderUnet):
    def set_net(self):
        inp = Input(shape=(*self.img_size, 3), name='raw')
        decoder = InceptionV3(include_top=False, input_tensor=inp)

        mixed_7_cropped = Cropping2D(((1,0),(1,0)))(decoder.get_layer('mixed7').output)
        encoder = conv_block(UpSampling2D()(decoder.output), 320)
        encoder = concatenate([encoder, mixed_7_cropped], axis=-1)
        encoder = conv_block(encoder, 320)

        mixed_2_cropped = Cropping2D(((2,1),(2,1)))(decoder.get_layer('mixed2').output)
        encoder = conv_block(UpSampling2D()(encoder), 256)
        encoder = concatenate([encoder, mixed_2_cropped], axis=-1)
        encoder = conv_block(encoder, 256)

        activ_5_cropped = Cropping2D(((4,3),(4,3)))(decoder.get_layer('activation_5').output)
        encoder = conv_block(UpSampling2D()(encoder), 128)
        encoder = concatenate([encoder, activ_5_cropped], axis=-1)
        encoder = conv_block(encoder, 128)

        activ_3_cropped = Cropping2D(((10,9),(10,9)))(decoder.get_layer('activation_3').output)
        encoder = conv_block(UpSampling2D()(encoder), 96)
        encoder = concatenate([encoder, activ_3_cropped], axis=-1)
        encoder = conv_block(encoder, 96)

        encoder = conv_block(UpSampling2D()(encoder), 64)
        encoder = conv_block(encoder, 64)
        res_mask = Conv2D(1, (1, 1), activation='softmax')(encoder)

        classification = Flatten(name='flatten')(decoder.output)
        classification = Dense(4096, activation='relu', name='fc1')(classification)
        res_classification = Dense(1, activation='sigmoid', name='predictions')(classification)

        model = Model(encoder, [res_mask, res_classification])

        self.neural_net = model
        self.decoder_layers = [(layer.name, layer.output_shape, i) for i, layer in enumerate(decoder.layers)]


def conv_block(prev, num_filters, kernel=(3, 3), strides=(1, 1), act='relu', prefix=None):
    name = None
    if prefix is not None:
        name = prefix + '_conv'
    conv = Conv2D(num_filters, kernel, padding='same', kernel_initializer='he_normal', strides=strides, name=name)(prev)
    if prefix is not None:
        name = prefix + '_norm'
    conv = BatchNormalization(name=name, axis=3)(conv)
    if prefix is not None:
        name = prefix + '_act'
    conv = Activation(act, name=name)(conv)
    return conv
