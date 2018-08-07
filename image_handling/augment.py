import random

import numpy as np
from skimage.transform import rotate


def augment(arrays):
    _arrays = (array.copy() for array in arrays)
    _arrays = _random_rotate(_arrays)
    _arrays = _random_flip(_arrays)

    return _arrays


def _random_rotate(arrays):
    rotations = random.randint(0, 3)
    for array in arrays:
        yield rotate(array, rotations*90)


def _random_flip(arrays):
    flip = random.choice([True, False])
    for array in arrays:
        yield np.fliplr(array) if flip else array
