from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, concatenate
from keras.layers import Cropping2D, Input

from .models import PretrainedDecoderRawEncoderUnet


class InceptionV3Unet(PretrainedDecoderRawEncoderUnet):
    def set_net(self):
        """
        build encoder-decoder network using the (pre-trained) InceptionV3 network.

        :return: None
        """
        # make input tensor based on numpy array
        inp = Input(shape=(*self.img_size, 3), name='image')
        # pre-trained encoder (inceptionv3)
        encoder = InceptionV3(include_top=False, input_tensor=inp)

        # first decoding block
        mixed_7_cropped = Cropping2D(((1,0),(1,0)))(encoder.get_layer('mixed7').output)
        decoder = conv_block(UpSampling2D()(encoder.output), 320)
        if self.skip_connections:
            decoder = concatenate([decoder, mixed_7_cropped], axis=-1)
        decoder = conv_block(decoder, 320)

        # second decoding block
        mixed_2_cropped = Cropping2D(((2,1),(2,1)))(encoder.get_layer('mixed2').output)
        decoder = conv_block(UpSampling2D()(decoder), 256)
        if self.skip_connections:
            decoder = concatenate([decoder, mixed_2_cropped], axis=-1)
        decoder = conv_block(decoder, 256)

        # third decoding block
        activ_5_cropped = Cropping2D(((4,3),(4,3)))(encoder.get_layer('activation_5').output)
        decoder = conv_block(UpSampling2D()(decoder), 128)
        if self.skip_connections:
            decoder = concatenate([decoder, activ_5_cropped], axis=-1)
        decoder = conv_block(decoder, 128)

        # fourth decoding block
        activ_3_cropped = Cropping2D(((10,9),(10,9)))(encoder.get_layer('activation_3').output)
        decoder = conv_block(UpSampling2D()(decoder), 96)
        if self.skip_connections:
            decoder = concatenate([decoder, activ_3_cropped], axis=-1)
        decoder = conv_block(decoder, 96)

        # fifth decoding block
        decoder = conv_block(UpSampling2D()(decoder), 64)
        decoder = conv_block(decoder, 64)

        # tensor with the output mask
        res_mask = Conv2D(self.mask_channels, (1, 1), activation='softmax', name='mask')(decoder)

        if self.classification:
            # build result tensor based on the output of the encoder
            res_classification = self._add_classification_branch(encoder)
            # model for multiple outputs (including the classification)
            model = Model([inp], [res_mask, res_classification])

        else:
            # model for single output (only the mask)
            model = Model([inp], [res_mask])

        # store model and meta-information regarding the layer names, shape, and index of the encoder layers
        self.neural_net = model
        self.encoder_layers = [(layer.name, layer.output_shape, i) for i, layer in enumerate(encoder.layers)]


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
