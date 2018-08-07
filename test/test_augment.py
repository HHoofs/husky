import numpy as np

from image_handling.augment import augment


def test_augment():
    array = np.array([[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]]).reshape((4,4,1))
    equal_arrays = (array, array)
    array_1, array_2 = augment(equal_arrays)

    assert np.array_equal(array_1, array_2), 'augmentation resulted arrays that are not equal'
