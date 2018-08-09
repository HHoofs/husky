"""juno

Usage:
    run.py (--vgg16|--inceptionv3) [-i <float>] [-cs] [-T]

Options:
    -h --help                          help
    --VGG16                            use the pretrained VGG16 encoder model
    --Inceptionv3                      use the pretrained InceptionV3 as encoder model
    -i <float> --Imagesize <float>     size of image to use in the model
    -c --Classification                add classifciation to the model
    -s --SkipConnections               add skip connections to the Unet model
    -T                                 test run


"""
from docopt import docopt


from models import inception_v3_unet, vgg16_unet
import collections
import random

import numpy as np
import csv
import os
from PIL import Image


def main(env_var):
    img_size = _pars_img_size(env_var)

    if env_var['--vgg16']:
        if img_size is None:
            img_size = (224, 224)
        model = vgg16_unet.vgg16_unet(img_size=img_size,
                                      classification=env_var['--Classification'],
                                      skip_connections=env_var['--SkipConnections'])
        model.set_net()
        model.freeze_encoder_blocks()
        model.neural_net.summary()

    if env_var['--inceptionv3']:
        pass


def _pars_img_size(env_var):
    img_size = env_var['--Imagesize']
    if img_size:
        img_size = (img_size, img_size)
    return img_size


if __name__ == '__main__':
    env_var = docopt(__doc__)
    print(env_var)
    main(env_var)
