"""
Module containing augmentations for CT data.
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

PROJ_ROOT = Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(str(PROJ_ROOT))
from project.tools.ndarray.random_crop import random_crop


def synth_augmentation(
    data: np.ndarray, name: str = ''
) -> Tuple[np.ndarray, str]:
    """
    Do some augmentation. Hard-written code just for tests right now.

    Parameters
    ----------
    data : np.ndarray
        Data that will be augmented.

    Returns
    -------
    Tuple[np.ndarray, str]
        Augmented sample.
    """
    # Some hard calculations just for time
    for _ in range(2):
        np.unique(data)

    return data, name


@tf.function
def tf_synth_augmentation(
    volume: tf.Tensor, name: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    TF wrapper for numpy function.

    This warpper is needed for including some function that operate
    with numpy arrays to TF computitional graph.

    Parameters
    ----------
    volume : tf.Tensor
        Tensor with CT volume.
    name : tf.Tensor
        Tensor with name of CT.

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor]
        Augmented sample.
    """
    volume, name = tf.numpy_function(synth_augmentation,
                                     [volume, name], (tf.float32, tf.string))
    return volume, name


def process_ct(
    volume: np.ndarray,
    crop_shape: Tuple[int],
    min_v: int = -1024,
    max_v: int = 3072,
    name: str = ''
) -> Tuple[np.ndarray, str]:
    """
    Do cropping, clipping and normalization for passed CT volume.

    Parameters
    ----------
    volume : np.ndarray
        CT volume that will be processed.
    crop_shape : Tuple[int]
        Shape of the volume crop.
    min_v : int, optional
        Min value for volumes (clip & norm).
    max_v : int, optional
        Max value for volumes (clip & norm).
    name : str, optional
        CT name. Need for pipeline.

    Returns
    -------
    Tuple[np.ndarray, str]
        Processed sample.
    """
    cropped_volume = random_crop(volume, crop_shape)
    cropped_volume = np.clip(cropped_volume, min_v, max_v)
    cropped_volume = ((cropped_volume - min_v) / (max_v - min_v))
    assert np.max(cropped_volume) <= 1
    return cropped_volume, name


@tf.function
def tf_process_ct(
    volume: tf.Tensor,
    crop_shape: Tuple[int],
    min_v: int = -1024,
    max_v: int = 3072,
    name: tf.Tensor = ''
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    TF wrapper for numpy function.

    This warpper is needed for including some function that operate
    with numpy arrays to TF computitional graph.

    Parameters
    ----------
    volume : tf.Tensor
        CT volume that will be processed.
    crop_shape : Tuple[int]
        Shape of the volume crop.
    min_v : int, optional
        Min value for volumes (clip & norm).
    max_v : int, optional
        Max value for volumes (clip & norm).
    name : tf.Tensor, optional
        CT name. Need for pipeline.

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor]
        Processed sample.
    """
    volume, name = tf.numpy_function(
        process_ct,
        [volume, crop_shape, min_v, max_v, name],
        (tf.float32, tf.string))
    return volume, name
