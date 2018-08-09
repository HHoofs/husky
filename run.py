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
import logging
from time import time

from docopt import docopt

from models import inception_v3_unet, vgg16_unet

logging.basicConfig(filename='logs/run.log', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(time())

def main(env_var):
    logger.info(env_var)
    img_size = _parse_img_size(env_var)

    if env_var['--vgg16']:
        if img_size is None:
            img_size = (224, 224)
        model_class = vgg16_unet.VGG16Unet

    elif env_var['--inceptionv3']:
        if img_size is None:
            img_size = (299, 299)
        model_class = inception_v3_unet.InceptionV3Unet

    else:
        return None

    model = model_class(img_size=img_size,
                        classification=env_var['--Classification'],
                        skip_connections=env_var['--SkipConnections'])
    model.set_net()
    model.freeze_encoder_blocks()
    model.neural_net.summary(print_fn=logger.info)


def _parse_img_size(env_var):
    img_size = env_var['--Imagesize']
    if img_size:
        img_size = (img_size, img_size)
    return img_size


if __name__ == '__main__':
    main(docopt(__doc__))
