"""Script that generates hdf5 file with synthetic data."""

import argparse
import os
import sys
from math import ceil
from pathlib import Path
from typing import Tuple

import cvl.hdf5 as chdf5
import h5py
import numpy as np
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
        '--path', help='Path for saving hdf5 file.', type=Path, required=True)

    parser.add_argument(
        '--number', help='Number of creating volumes', type=natural_int,
        default=100)
    parser.add_argument(
        '--shape', help='Shape of volumes.', type=natural_int, nargs=3,
        default=(256, 256, 256))
    parser.add_argument(
        '--buf-size', help='Size of dump buffer.', type=natural_int,
        default=10)

    args = parser.parse_args()

    return args


def create_synth_hdf5(path: Path, number: int,
                      shape: Tuple[int, int, int],
                      buf_size: int = 1) -> None:
    """
    Create and fill hdf5 with synthetic data.
    
    Generates n random volumes of type int16 in the range [-4096, 4096]
    and writes it all to a hdf5 file.

    Parameters
    ----------
    path : Path
        Path to the out hdf5 file.
    number : int
        Number of volumes that will be generated.
    shape : Tuple[int, int, int]
        Shape of generating volumes.
    buf_size : int, optional
        Size of loading buffer, by default 1.
    """
    dataset_desk = {
        'volumes': chdf5.DatasetDescription(shape, dtype=np.int16),
        'names': chdf5.DatasetDescription((), dtype=h5py.string_dtype())
    }
    attributes = {
        'size': number,
        'shape': shape
    }

    logger.info(f'Created h5 file: "{path}".')
    file = chdf5.init_storage(str(path), attrs=attributes, **dataset_desk)
    name_title = os.path.split(path)[-1].split('.')[0] + ' sample '
    dump_number = ceil(number / buf_size)
    indx = 0
    logger.info(f'Start generation with buffer size = "{buf_size}"...')
    for _ in tqdm(range(dump_number)):
        sample_buf = np.empty((buf_size, *shape))
        names = []
        for i in range(buf_size):
            logger.debug('Generate...')
            sample_buf[i] = np.random.randint(-4096, 4096, size=shape)
            names.append(name_title + str(indx))
            logger.debug('OK.')
            indx += 1
        logger.debug('Dump...')
        chdf5.store_data(file, **{'volumes': sample_buf,
                                  'names': names})
        logger.debug('OK.')
    file.close()
    logger.info('OK.')


def main():
    """Application entry point."""
    args = parse_args()
    create_synth_hdf5(args.path, args.number, tuple(args.shape),
                      args.buf_size)


if __name__ == '__main__':
    main()
