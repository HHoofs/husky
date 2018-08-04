from models import inception_v3_unet, vgg16_unet
import collections
import random

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from PIL import Image


def main():
    model = inception_v3_unet.inception_v3_unet()
    model = inception_v3_unet.InceptionV3Unet((299,299))
    model.summary()


if __name__ == '__main__':
    main()