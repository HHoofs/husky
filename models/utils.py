from keras.callbacks import Callback
from keras_applications.vgg16 import preprocess_input as vgg16_pre
from keras_applications.inception_v3 import preprocess_input as inception_v3_pre


def select_pre_processing(encoder_model):
    if encoder_model == 'VGG16':
        return vgg16_pre
    elif encoder_model == 'InceptionV3':
        return inception_v3_pre
    elif encoder_model == 'Default':
        return vgg16_pre
    else:
        assert False, 'No valid encoder model is provided to the generator'
