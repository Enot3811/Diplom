import sys
import argparse
from pathlib import Path
from typing import Callable

import h5py
import tensorflow as tf


PROJ_ROOT = Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(str(PROJ_ROOT))
from project.pipeline.h5_reading.samplers import PureCTSampler
from project.pipeline.tfrecord_reading.deserialization import (
    deserialize_example)
from visualizers import slice_visualize, aggregation_visualize


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--path',
        help='Path to dataset (h5 file or directory with tfrecords).',
        type=Path,
        required=True)
    parser.add_argument(
        '--visualization-type',
        help='Type of implemented CT visualization.',
        choices=['Aggregation', 'Slices'],
        default='Aggregation',
        type=str)

    args = parser.parse_args()

    if not args.path.exists():
        raise FileNotFoundError(
            f'Specified dataset path "{args.path}" does not exist.')

    return args


def visualize_h5_dset(dset_path: Path, visualizator: Callable):
    """
    Visualize h5 dataset with specified visualizator function.

    Parameters
    ----------
    dset_path : Path
        Path to h5 dataset.
    visualizator : Callable
        Visualizator function.
    """
    sampler = PureCTSampler()
    with h5py.File(dset_path, 'r') as f:
        size = f.attrs['size']
    gen = sampler(str(dset_path), 0, size)
    for ct, name in gen:
        visualizator(ct, title=name)


def visualize_tfrec_dset(dset_path: Path, visualizator: Callable):
    """
    Visualize h5 dataset with specified visualizator function.

    Parameters
    ----------
    dset_path : Path
        Path to directory with tfrecord dataset.
    visualizator : Callable
        Visualizator function.
    """
    file_paths = [str(file.absolute()) for file in dset_path.glob('*')]
    paths_dset = tf.data.Dataset.from_tensor_slices(file_paths)
    raw_dataset = tf.data.TFRecordDataset(paths_dset)
    dset = raw_dataset.map(deserialize_example).as_numpy_iterator()
    for ct, name in dset:
        visualizator(ct, title=name.decode())


def main():
    """Application entry point."""
    args = parse_args()

    visualize_functions = {
        'Aggregation': aggregation_visualize,
        'Slices': slice_visualize
    }
    visualizator = visualize_functions[args.visualization_type]

    if args.path.is_dir():
        visualize_tfrec_dset(args.path, visualizator)
    elif args.path.is_file():
        visualize_h5_dset(args.path, visualizator)
    else:
        raise ValueError(
            f'Specified dataset path "{args.path}" does not correspond '
            f'neither HDF5 nor TFRecord dataset.')


if __name__ == '__main__':
    main()
