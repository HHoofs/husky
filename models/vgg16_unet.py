import keras
import keras.backend as K
import numpy as np
from keras import models, Input, Model
from keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Dense, concatenate, MaxPooling2D, \
    Concatenate, AveragePooling2D, Cropping2D, Flatten
from keras_applications.vgg16 import VGG16

def vgg16_unet(img_size=(224, 224)):
    inp = Input(shape=(*img_size, 3), name='raw')
    decoder = VGG16(include_top=False, input_tensor=inp)

    encoder = UpSampling2D()(decoder.get_layer('block5_conv3').output)
    encoder = concatenate([encoder, decoder.get_layer('block4_conv3').output], axis=-1)
    encoder = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder)
    encoder = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder)

    encoder = UpSampling2D()(encoder)
    encoder = concatenate([encoder, decoder.get_layer('block3_conv3').output], axis=-1)
    encoder = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder)
    encoder = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder)

    encoder = UpSampling2D()(encoder)
    encoder = concatenate([encoder, decoder.get_layer('block2_conv2').output], axis=-1)
    encoder = Conv2D(64, (3, 3), activation='relu', padding='same')(encoder)
    encoder = Conv2D(64, (3, 3), activation='relu', padding='same')(encoder)

    encoder = UpSampling2D()(encoder)
    encoder = concatenate([encoder, decoder.get_layer('block1_conv2').output], axis=-1)
    encoder = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder)
    encoder = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder)
    res_mask = Conv2D(1, (1, 1), activation='softmax')(encoder)

    classification = Flatten(name='flatten')(decoder.output)
    classification = Dense(4096, activation='relu', name='fc1')(classification)
    res_classification = Dense(1, activation='sigmoid', name='predictions')(classification)
