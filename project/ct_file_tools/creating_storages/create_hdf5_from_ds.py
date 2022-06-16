import sys
import argparse
import gc
from pathlib import Path
from typing import Callable, Dict, Any

import h5py
from loguru import logger
from cvl import hdf5

PROJ_ROOT = Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(str(PROJ_ROOT))
from project.tools.argparse.types import natural_int
from project.ct_file_tools.loaders.dicom_reader import load_dicom
from project.ct_file_tools.loaders.nifti_reader import load_nifti


def parse_args() -> argparse.Namespace:
    """Create and parse command-line arguments.
    
    Returns
    -------
    argparse.Namespace
        Arguments from command line.

    Raises
    ------
    FileNotFoundError
        Could not find source dir.
    ValueError
        Dump buffer size must be positive.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--source-dir',
        type=Path,
        required=True,
        help='Path to dataset dir.')

    parser.add_argument(
        '--out-hdf5',
        type=Path,
        required=True,
        help='Path for out hdf5 file.')

    parser.add_argument(
        '--file-extension',
        type=str,
        choices=['dicom', 'nifti'],
        required=True,
        help='Extension of read CT files.')

    parser.add_argument(
        '--dump-buffer-size',
        type=natural_int,
        default=1,
        help='Number of elements that will be written at once.')

    args = parser.parse_args()

    if not args.source_dir.is_dir():
        raise FileNotFoundError(f'Dataset directory {args.source_dir} '
                                'does not exist.')

    if args.out_hdf5.is_file():
        logger.warning(
            f'Hdf5 file {args.out_hdf5} already exists. '
            f'Press enter to continue.'
        )
        input()

    return args


def create_hdf5(
    source_dir: Path,
    h5_path: Path,
    loader: Callable,
    dump_buffer_size: int = 1
):
    """
    Read CT files and save its data to hdf5.

    Parameters
    ----------
    source_dir : Path
        Path to root dir that contain CT files.
    h5_path : Path
        Path to save hdf5 file.
    loader : Callable
        Loader for extension of loading files.
    dump_buffer_size : int, optional
        Number of elements that will be written at once, by default 1.
    """
    ds_descriptions = {
        'volumes': hdf5.DatasetDescription(
            shape=(), dtype=h5py.vlen_dtype('int16')
        ),
        'shapes': hdf5.DatasetDescription(
            shape=(3,), dtype='uint16'
        ),
        'names': hdf5.DatasetDescription(
            shape=(), dtype=h5py.string_dtype()
        )
    }
    
    h5_file = hdf5.init_storage(str(h5_path), file_mode='w', **ds_descriptions)

    buffer: Dict[str, Any] = {k: [] for k in list(h5_file)}

    buf_size = 0
    ds_size = 0
    for ct_sample in source_dir.glob('*'):
        ct_volume = loader(ct_sample)

        buffer['shapes'].append(ct_volume.shape)
        buffer['volumes'].append(ct_volume.flatten())
        buffer['names'].append(ct_sample.name)

        buf_size += 1
        ds_size += 1

        logger.debug(f'Buffer: {buf_size} - Dataset: {ds_size}')

        if buf_size == dump_buffer_size:
            hdf5.store_data(h5_file, **buffer)
            buffer = {k: [] for k in buffer.keys()}
            buf_size = 0
            gc.collect()
            h5_file.flush()
            logger.debug('Dump.')
    
    if buf_size > 0:
        hdf5.store_data(h5_file, **buffer)
        h5_file.flush()
        logger.debug('Dump.')

    h5_file.attrs['size'] = ds_size
    h5_file.close()


def main():
    """Application entry point."""
    args = parse_args()

    loaders = {
        'dicom': load_dicom,
        'nifti': load_nifti
    }

    create_hdf5(args.source_dir,
                args.out_hdf5,
                loaders[args.file_extension],
                args.dump_buffer_size)


if __name__ == '__main__':
    main()
