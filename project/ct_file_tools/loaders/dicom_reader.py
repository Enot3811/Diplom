"""Module for loading dicom files."""


from pathlib import Path

import pydicom as pdcm
import numpy as np


def load_dicom(source_dir: Path) -> np.ndarray:
    """
    Read CT from dicom files.

    Sequentially reads CT slices from dicom files in specified directory
    and concatenate them to single volume.

    Parameters
    ----------
    source_dir : Path
        Path to dir containing dicom files.

    Returns
    -------
    np.ndarray
       Volume collected from slices.

    Raises
    ------
    FileNotFoundError
        Specified dir does not exist.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f'Specified dir {source_dir} does not exist.')
    files = source_dir.glob('*.dcm')
    first_dcm = True
    for file in files:
        ds = pdcm.read_file(file)
        # Compile 2d from dicom files to 3d volume
        if first_dcm:
            volume = ds.pixel_array
            volume = volume[..., None]
            first_dcm = False
        else:
            volume = np.concatenate(
                (volume, ds.pixel_array[..., None]), axis=2)

    return volume
