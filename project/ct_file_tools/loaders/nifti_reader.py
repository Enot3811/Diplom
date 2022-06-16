"""Module for loading NIfTI files."""


from pathlib import Path

import numpy as np
import nibabel as nib


def load_nifti(source_path: Path) -> np.ndarray:
    """
    Read CT from NIfTI file.
    """
    if not source_path.is_file():
        raise FileNotFoundError(
            f'Specified file {source_path} does not exist.')
    ct_volume = nib.load(source_path).get_fdata()
    return ct_volume
