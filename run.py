"""Husky

Usage:
    run.py (--VGG16|--InceptionV3) (--Binary|--Categorical) [-i <float>] [-cs] [-T] [-e <float>]

Options:
    -h --help                          help
    --VGG16                            use the pre-trained VGG16 encoder model
    --InceptionV3                      use the pre-trained InceptionV3 as encoder model
    --Binary                           binary mask (mask or background)
    --Categorical                      categorical mask (border, mask, background)
    -i <float> --ImageSize <float>     size of image to use in the model
    -c --Classification                add classification to the model
    -s --SkipConnections               add skip connections to the U-net model
    -e <float> --Epochs <float>        number of epochs for the training [default: 1]
    -T                                 test run


"""
import logging
import time

from docopt import docopt
from keras import losses

from file_handling.read_db import read_csv_encoding, ids_for_each_count
from generator import DataGenerator
from models import inception_v3_unet, vgg16_unet
from models.models import iou_calc_second, iou_calc_first

logging.basicConfig(filename='logs/{}_run.log'.format(time.strftime("%Y%m%d_%H%M%S")),
                    level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(time.strftime("%Y%m%d_%H%M%S"))


def main(env_var):
    logger.info(env_var)
    img_size, epochs, mask_channels, mask_type, metric_sel, loss_sel = _parse_env_var(env_var)

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

    image_encoding = read_csv_encoding(length=1280)

    ids = list(image_encoding.keys())

    training_gen = DataGenerator(ids=ids[:1200],
                                 img_encodings=image_encoding,
                                 mask_type=mask_type,
                                 out_dim_img=img_size,
                                 classification=env_var['--Classification'])

    validati_gen = DataGenerator(ids=ids[1200:1280],
                                 img_encodings=image_encoding,
                                 mask_type=mask_type,
                                 out_dim_img=img_size,
                                 classification=env_var['--Classification'])

    model = model_class(img_size=img_size,
                        classification=env_var['--Classification'],
                        skip_connections=env_var['--SkipConnections'],
                        mask_channels=mask_channels)
    model.set_net()
    model.freeze_encoder_blocks()
    model.compile(loss=loss_sel, metrics=metric_sel)
    model.neural_net.summary(print_fn=logger.info)
    model.fit(training_generator=training_gen, validation_generator=validati_gen,
              epochs=epochs)

    predicti_gen = DataGenerator(ids=ids[:8],
                                 img_encodings=image_encoding,
                                 mask_type=mask_type,
                                 out_dim_img=img_size,
                                 classification=env_var['--Classification'],
                                 shuffle=False)

    model.predict(pred_generator=predicti_gen)


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

    # classification
    if env_var['--Classification']:
        _metrics = {'mask': None, 'classification': 'acc'}
        _losses = {'mask': None, 'classification': losses.binary_crossentropy}
    else:
        _metrics = {'mask': None}
        _losses = {'mask': None}

    # mask type
    if env_var['--Categorical']:
        mask_type = 'categorical'
        mask_channels = 3
        _metrics['mask'] = iou_calc_second
        _losses['mask'] = losses.categorical_crossentropy

    elif env_var['--Binary']:
        mask_type = 'binary'
        mask_channels = 1
        _metrics['mask'] = iou_calc_first
        _losses['mask'] = losses.binary_crossentropy
    else:
        assert False, 'no valid mask type provided'

    return img_size, epochs, mask_channels, mask_type, _metrics, _losses


if __name__ == '__main__':
    main(docopt(__doc__))
