from keras import Input, Model
from keras.layers import Conv2D, UpSampling2D, concatenate
from keras_applications.vgg16 import VGG16

from .models import PretrainedDecoderRawEncoderUnet


class VGG16Unet(PretrainedDecoderRawEncoderUnet):
    def set_net(self):
        """
        build encoder-decoder network using the (pre-trained) VGG16 network.

        :return: None
        """
        # make input tensor based on numpy array
        inp = Input(shape=(*self.img_size, 3), name='image')
        # pre-trained encoder (vgg16)
        encoder = VGG16(include_top=False, input_tensor=inp)

        # first decoding block
        decoder = UpSampling2D()(encoder.get_layer('block5_conv3').output)
        if self.skip_connections:
            decoder = concatenate([decoder, encoder.get_layer('block4_conv3').output], axis=-1)
        decoder = Conv2D(256, (3, 3), activation='relu', padding='same')(decoder)
        decoder = Conv2D(256, (3, 3), activation='relu', padding='same')(decoder)

        # second decoding block
        decoder = UpSampling2D()(decoder)
        if self.skip_connections:
            decoder = concatenate([decoder, encoder.get_layer('block3_conv3').output], axis=-1)
        decoder = Conv2D(128, (3, 3), activation='relu', padding='same')(decoder)
        decoder = Conv2D(128, (3, 3), activation='relu', padding='same')(decoder)

        # third decoding block
        decoder = UpSampling2D()(decoder)
        if self.skip_connections:
            decoder = concatenate([decoder, encoder.get_layer('block2_conv2').output], axis=-1)
        decoder = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder)
        decoder = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder)

        # fourth decoding block
        decoder = UpSampling2D()(decoder)
        if self.skip_connections:
            decoder = concatenate([decoder, encoder.get_layer('block1_conv2').output], axis=-1)
        decoder = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder)
        decoder = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder)

        # tensor with the output mask
        res_mask = Conv2D(3, (1, 1), activation='softmax', name='mask')(decoder)

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
