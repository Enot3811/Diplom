"""
Samplers for HDF5 files.
"""

import sys
from math import ceil
from pathlib import Path
from typing import Tuple
from abc import ABC, abstractmethod
from typing import Generator

import h5py
import numpy as np
import cvl.hdf5 as chdf5

PROJ_ROOT = Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(str(PROJ_ROOT))
from project.tools.ndarray.random_crop import random_crop

VolumeShapeType = Tuple[int, int, int]


class H5Sampler(ABC):
    """Abstract class for buffered dataset generator."""

    SampleType = Tuple[np.ndarray, str]
    GeneratorType = Generator[SampleType, None, None]

    def __init__(self, buffer_size: int = 1):
        """Get batch size and save it in self.

        Parameters
        ----------
        buffer_size : int, optional
            Number of samples that will be loaded at one time.
        """
        self.buffer_size = buffer_size

    @abstractmethod
    def __call__(self, h5_path: str, start: int, end: int) -> GeneratorType:
        """Do something when generator is called.

        Parameters
        ----------
        h5_path : str
            The path to hdf5 file with data for the dataset.
        start : int
            The Index of sample from which generator starts reading.
        end : int
            The index of sample to which the generator reads.
        """
        pass


class PureCTSampler(H5Sampler):
    """
    Functor-generator that performs 3D-CT-Scans.

    Function inits with static params of generation.
    Each call creates python-generator object that yields necessary data in
    specified range.
    """
    
    def __call__(
        self, h5_path: str,
        start: int, end: int
    ) -> H5Sampler.GeneratorType:
        """
        Generate batch of CT.

        Load several CT volumes and yield them one by one.

        Parameters
        ----------
        h5_path : str
            The path to hdf5 file with data for the dataset.
        start : int
            The Index of sample from which generator starts reading.
        end : int
            The index of sample to which the generator reads.

        Yields
        ------
        H5Generator.GeneratorType[SampleType]:
            Loads the next batch of samples contained pair with volume and
            its name.
        """
        # Recalculate start & end indices related to batches necessary to read
        end = ceil(end / self.buffer_size)
        start = ceil(start / self.buffer_size)

        with h5py.File(h5_path, 'r') as f:
            data_arrs = list(f.keys())
        load_keys = ['volumes', 'names']
        if 'shapes' in data_arrs:
            load_keys.append('shapes')
            need_reshape = True
        else:
            need_reshape = False
        data_loader = chdf5.BatchLoader(self.buffer_size, h5_path, load_keys)

        for batch_indx in range(start, end, 1):
            batch = data_loader.load(batch_indx)
            volumes = batch[0]
            names = [name.decode() for name in batch[1]]
            if need_reshape:
                shapes = batch[2]
            else:
                shapes = np.array([x.shape for x in volumes])

            for i in range(self.buffer_size):
                volume = np.reshape(volumes[i], shapes[i])
                yield volume, names[i]


class CTSampler(H5Sampler):
    """
    Functor-generator that performs 3D-CT-Scans random crops sampling.

    Function inits with static params of generation.
    Each call creates python-generator object that yields necessary data in
    specified range.
    """

    def __init__(self, crop_shape: VolumeShapeType, buffer_size: int = 1,
                 min_v: int = -1024, max_v: int = 3072):
        """
        Create instance of the functor-generator.

        Parameters
        ----------
        crop_shape : VolumeShapeType
            Shape of the volume crop.
        min_v : int
            Min value for volumes (clip & norm).
        max_v : int
            Max value for volumes (clip & norm).
        """
        super().__init__(buffer_size)
        self.crop_shape = crop_shape
        self.min_v = min_v
        self.max_v = max_v

    def __call__(
        self, h5_path: str,
        start: int, end: int
    ) -> H5Sampler.GeneratorType:
        """
        Generate batch of CT 3d crops.

        1) Load several CT volumes
        2) Crop Each
        3) Yield

        Parameters
        ----------
        h5_path : str
            The path to hdf5 file with data for the dataset.
        start : int
            The Index of sample from which generator starts reading.
        end : int
            The index of sample to which the generator reads.

        Yields
        ------
        H5Generator.GeneratorType[SampleType]:
            Loads the next batch of samples contained pair with volume and
            its name.
        """
        # Recalculate start & end indices related to batches necessary to read
        end = ceil(end / self.buffer_size)
        start = ceil(start / self.buffer_size)

        cropped_volumes = np.empty((self.buffer_size, *self.crop_shape),
                                   dtype=np.int16)

        with h5py.File(h5_path, 'r') as f:
            data_arrs = list(f.keys())
        load_keys = ['volumes', 'names']
        if 'shapes' in data_arrs:
            load_keys.append('shapes')
            need_reshape = True
        else:
            need_reshape = False
        data_loader = chdf5.BatchLoader(self.buffer_size, h5_path, load_keys)

        for batch_indx in range(start, end, 1):
            batch = data_loader.load(batch_indx)
            volumes = batch[0]
            names = [name.decode() for name in batch[1]]
            if need_reshape:
                shapes = batch[2]
            else:
                shapes = np.array([x.shape for x in volumes])

            for i in range(self.buffer_size):
                volume = np.reshape(volumes[i], shapes[i])
                cropped_volumes[i] = random_crop(volume, self.crop_shape)

            cropped_volumes = cropped_volumes.astype(np.float32)
            cropped_volumes = np.clip(cropped_volumes, self.min_v, self.max_v)
            cropped_volumes = ((cropped_volumes - self.min_v) /
                               (self.max_v - self.min_v))
            assert np.max(cropped_volumes) <= 1
            yield cropped_volumes, names


class CTAugmentedSampler(CTSampler):
    """
    Functor-generator that performs 3D-CT-Scans random crops sampling.

    The main difference from base class: applying augmentations.

    Function inits with static params of generation.
    Each call creates python-generator object that yields necessary data in
    specified range.
    """

    def __init__(self, crop_shape: VolumeShapeType, buffer_size: int = 1,
                 min_v: int = -1024, max_v: int = 3072):
        """
        Create instance of the functor-generator.

        Parameters
        ----------
        crop_shape : VolumeShapeType
            Shape of the volume crop.
        """
        super().__init__(crop_shape, buffer_size, min_v, max_v)

    def __call__(
        self, h5_path: str, start: int, end: int
    ) -> H5Sampler.GeneratorType:
        """
        Create python-generator that yields random-crop of CT augmented data.

        1) Load several CT volumes
        2) Crop Each
        3) Yield

        Sampling is performed in ``[start, end)``.

        Parameters
        ----------
        h5_path : str
            The path to hdf5 file with data for the dataset.
        start : int
            The Index of sample from which generator starts reading.
        end : int
            The index of sample to which the generator reads.
            Not included.

        Yields
        ------
        H5Generator.GeneratorType[SampleType]:
            Loads the next CT-3D-sample: CT-scan & CT-name.
        """
        gen = super().__call__(h5_path, start, end)
        for read_data in gen:
            crop, name = read_data
            crop = self._do_augmentation(crop)
            yield crop, name

    @staticmethod
    def _do_augmentation(data: np.ndarray) -> np.ndarray:
        """
        Do some augmentation. Hard-written code just for tests right now.

        Parameters
        ----------
        data : np.ndarray
            Data that will be augmented.

        Returns
        -------
        np.ndarray
            Augmented data.
        """
        # Some hard calculations just for time
        for _ in range(2):
            np.unique(data)

        return data
