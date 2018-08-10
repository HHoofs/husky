"""Husky

Usage:
    run.py (--VGG16|--InceptionV3) [-i <float>] [-cs] [-T] [-e <float>]

Options:
    -h --help                          help
    --VGG16                            use the pre-trained VGG16 encoder model
    --InceptionV3                      use the pre-trained InceptionV3 as encoder model
    -i <float> --ImageSize <float>     size of image to use in the model
    -c --Classification                add classification to the model
    -s --SkipConnections               add skip connections to the U-net model
    -e <float> --Epochs <float>        number of epochs for the training [default: 1]
    -T                                 test run


"""
import logging
import time

from docopt import docopt

from file_handling.read_db import read_csv_encoding
from generator import DataGenerator
from models import inception_v3_unet, vgg16_unet

logging.basicConfig(filename='logs/{}_run.log'.format(time.strftime("%Y%m%d_%H%M%S")),
                    level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(time.strftime("%Y%m%d_%H%M%S"))


def main(env_var):
    logger.info(env_var)
    img_size, epochs = _parse_env_var(env_var)

    if env_var['--VGG16']:
        if img_size is None:
            img_size = (224, 224)
        model_class = vgg16_unet.VGG16Unet

    elif env_var['--InceptionV3']:
        if img_size is None:
            img_size = (299, 299)
        model_class = inception_v3_unet.InceptionV3Unet

    else:
        return None

    image_encoding = read_csv_encoding(length=50)

    training_gen = DataGenerator(img_encodings=image_encoding,
                                 out_dim_img=img_size,
                                 classification=env_var['--Classification'])

    model = model_class(img_size=img_size,
                        classification=env_var['--Classification'],
                        skip_connections=env_var['--SkipConnections'])
    model.set_net()
    model.freeze_encoder_blocks()
    model.compile(metrics=['acc'])
    model.neural_net.summary(print_fn=logger.info)
    model.fit(training_generator=training_gen,
              epochs=epochs)


def _parse_env_var(env_var):
    # image size
    img_size = env_var['--ImageSize']
    if img_size:
        img_size = (img_size, img_size)

    # epochs
    try:
        epochs = int(env_var['--Epochs'])
    except ValueError:
        logger.error('The command line argument for --Epochs should be an integer')
        return

    return img_size, epochs


if __name__ == '__main__':
    main(docopt(__doc__))
