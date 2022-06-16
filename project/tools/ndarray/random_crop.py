# -*- coding: UTF-8 -*-

"""Random crop of the N-dim numpy.ndarray_ext."""

from typing import Tuple

import numpy as np


def random_crop(
    data: np.ndarray,
    crop_shape: Tuple[int, ...]
) -> np.ndarray:
    """
    Crop given data to given shapes.

    Crop can be done only in the area where requested shape will exist.
    No padding is added.

    Parameters
    ----------
    data : np.ndarray
        Volume that will be cropped.
    crop_shape : Tuple[int]
        Shape for cropped data.

    Returns
    -------
    np.ndarray
        Cropped volume.

    Raises
    ------
    ValueError
        The dimension of the crop does not match the dimension of the data
    ValueError
        'One of dimensions of crop shape is bigger than the original one'
    """
    init_shape = data.shape

    if len(init_shape) != len(crop_shape):
        raise ValueError(
            'The dimension of the crop does not match' +
            'the dimension of the data'
        )
    if np.any(np.array(init_shape) < np.array(crop_shape)):
        raise ValueError(
            'One of dimensions of crop shape is bigger than the original one'
        )
    if crop_shape is not tuple:
        crop_shape = tuple(crop_shape)

    slices = []
    for i in range(len(crop_shape)):
        random_point = np.random.randint(0, init_shape[i] - crop_shape[i] + 1)
        slices.append(slice(random_point, random_point + crop_shape[i]))
    crop = data[tuple(slices)]
    assert crop.shape == crop_shape
    return crop
