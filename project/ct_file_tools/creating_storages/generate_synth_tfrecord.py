"""Script that generates tfrecord files with synthetic data."""

import argparse
import sys
import os
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import tensorflow as tf
from loguru import logger
from tqdm import tqdm

PROJ_ROOT = Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(str(PROJ_ROOT))
from project.tools.argparse.types import natural_int


def parse_args() -> argparse.Namespace:
    """Create & parse arguments from the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    out_args = parser.add_argument_group('Outputs')
    out_args.add_argument(
        '--path', help='Path for saving tfrecord files.',
        type=Path, required=True)

    parser.add_argument(
        '--number', help='Number of creating volumes', type=natural_int,
        default=100)
    parser.add_argument(
        '--shape', help='Shape of volumes.', type=natural_int, nargs=3,
        default=(256, 256, 256))
    parser.add_argument(
        '--num-files', help='Number of creating files', type=natural_int,
        default=1)

    args = parser.parse_args()

    return args


def bytes_feature(value: Union[tf.Tensor, np.ndarray]) -> tf.train.Feature:
    """
    Returns a bytes_list from a string / byte.

    Parameters
    ----------
    value : Union[tf.Tensor, np.ndarray]
        Data volume.

    Returns
    -------
    tf.train.Feature
        Serialized to bytes data.
    """
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(
    volume: Union[tf.Tensor, np.ndarray],
    name: Union[str, bytes]
) -> bytes:
    """
    Serialize data before writing to tfrecord.

    Parameters
    ----------
    volume : Union[tf.Tensor, np.ndarray]
        Tensor or ndarray with data volume.
    name : Union[str, bytes]
        Name of sample.

    Returns
    -------
    bytes
        Serialized data.
    """
    # Create a dictionary mapping the feature name
    # to the tf.train.Example-compatible data type.
    volume = tf.io.serialize_tensor(volume)

    # Can serialize bytes, not string
    if isinstance(name, str):
        name = bytes(name, 'utf-8')

    feature = {
        'volume': bytes_feature(volume),
        'name': bytes_feature(name)
    }
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    # Serialize example to byte string
    return example_proto.SerializeToString()


def tf_serialize_example(
    volume: tf.Tensor,
    name: tf.Tensor
) -> tf.Tensor:
    """
    TF function wrapper for serialize_example.

    Parameters
    ----------
    volume : tf.Tensor
        Tensor contained data volume.
    name : tf.Tensor
        Tensor contained byte string with sample name.

    Returns
    -------
    tf.Tensor
        Tensor contained serialized sample.
    """
    tf_string = tf.py_function(
        serialize_example,
        (volume, name),
        tf.string)
    return tf.reshape(tf_string, ())  # The result is a scalar.


def create_synth_tfrecord(path: Path,
                          number: int,
                          shape: Tuple[int, int, int],
                          num_files: int):
    """
    Generate tfrecord files contained synth data.

    Parameters
    ----------
    path : Path
        Path to the out dir with tfrecord files.
    number : int
        Number of volumes that will be generated.
    shape : Tuple[int, int, int]
        Shape of generating volumes.
    num_files : int
        Number of files over which the data will be distributed.
    """
    os.makedirs(path, exist_ok=True)
    file_name = os.path.split(path)[-1]
    name_title = file_name + ' sample'
    indx = 0

    logger.info('Start generation...')

    boundaries = np.array(np.around(np.linspace(0, number, num_files + 1)),
                          dtype=np.int32)

    paths = [os.path.join(path, f'{file_name} #{i}.tfrecord')
             for i in range(num_files)]
    for i, file_path in enumerate(paths):
        logger.debug(f'Generate {i + 1}-th file...')
        with tf.io.TFRecordWriter(str(file_path)) as writer:
            for _ in tqdm(range(boundaries[i], boundaries[i + 1])):
                volume = tf.random.uniform(shape)
                name = bytes(f'{name_title} {indx}', 'utf-8')
    
                example = serialize_example(volume, name)
                writer.write(example)
    
                indx += 1
        logger.debug('Done.')
    logger.debug('OK.')


def main():
    """Application entry point."""
    args = parse_args()
    create_synth_tfrecord(args.path, args.number,
                          tuple(args.shape), args.num_files)


if __name__ == '__main__':
    main()
