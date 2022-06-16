"""The module with the split managers for CT dataset."""

import sys
from typing import Callable, Tuple, Dict, List, Union
from pathlib import Path
from math import ceil

import h5py

PROJ_ROOT = Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(str(PROJ_ROOT))
from project.pipeline.h5_reading.samplers import H5Sampler


class H5SplitManager(object):
    """
    Split manager for CT datasets.
    This class is for splitting and loading data from h5 file(s).
    """

    class SingleReader:
        """
        Reader class for single dataset.
        
        Reader is needed for referring to a certain part of data
        in hdf5 files (now this part is separate dataset)
        and returning generator that will load this data.
        """

        def __init__(
            self, h5_path: str, start: int, end: int, generator: Callable
        ):
            """
            Initialize SingleReader object
            
            During the initialization object saves given parameters in self.

            Parameters
            ----------
            h5_path : str
                The path to hdf5 file with data.
            start : int
                The index of sample in file from which loading will start.
            end : int
                The index of sample in file at which loading will end.
            generator : Callable
                The generator functor for loading data from file.
            """
            self.h5_path = h5_path
            self.start = start
            self.end = end
            self.generator = generator

        def call(self) -> H5Sampler.GeneratorType:
            """
            Return generator for reading dataset.
            
            Calls the saved generator functor with put
            other saved parameters into him as a return.

            Returns
            -------
            H5Generator.GeneratorType:
                Saved generator for loading dataset.
            """
            return self.generator(self.h5_path, self.start, self.end)

    def __init__(self,
                 dataset_configs: List[Dict[str, Union[str, int]]],
                 generators: Dict[str, H5Sampler]):
        """
        Initialize `SingleReader` object.

        During the initialization each file from given config is split
        into specify number of splits and then each resulted part
        of data from file is put to `SingleReader` as a path to ``hdf5`` file,
        start/end indexes and a generator for loading.

        At the end of the initialization, the split-manager object will store
        a list of `SingleReader`, each of which will refer to a certain part
        of the data from the specified file.

        Parameters
        ----------
        dataset_configs : List[Dict[str, Union[str, int]]]
            List with dataset settings dictionaries contained
            paths to ``hdf5`` data files, numbers of split for each file
            and generator type for loading from corresponding file.
        generators : Dict[str, H5Generator]
            Dict with different types of initialized generator objects.
        """
        self._readers = []

        for dset_conf in dataset_configs:
            
            generator_name = dset_conf['generator_name']
            h5_path = dset_conf['h5_path']
            split_number = dset_conf['split_number']

            with h5py.File(h5_path, 'r') as f:
                dataset_len = f.attrs['size']

            # Get list of start/end indexes
            boundaries = H5SplitManager.split_dataset(dataset_len,
                                                      split_number)

            # Create reader for each split
            for (start, end) in boundaries:
                self._readers.append(
                    H5SplitManager.SingleReader(h5_path,
                                                start,
                                                end,
                                                generators[generator_name]))
    
    def __getitem__(self, i: int) -> H5Sampler.GeneratorType:
        """
        Get generator for i-th dataset by calling corresponding reader.

        Parameters
        ----------
        i : int
            Index of requested dataset.

        Returns
        -------
        H5Sampler.GeneratorType
            Generator for requested dataset.
        """
        return self._readers[i].call()

    def __len__(self) -> int:
        """
        Get number of datasets (number of readers).

        Returns
        -------
        int
            Number of datasets.
        """
        return len(self._readers)

    @staticmethod
    def split_dataset(dataset_length: int,
                      split_number: int) -> List[Tuple[int, int]]:
        """
        Split dataset into several parts represented as start, end index.

        Parameters
        ----------
        dataset_length : int
            Length of dataset that will splitted.
        split_number : int
            Number of dataset splits needed.

        Returns
        -------
        List[Tuple[int, int]]
            List with tuples contained start/end indexes.
        """
        split_step = ceil(dataset_length / split_number)
        start_positions = list(range(0, dataset_length, split_step))
        end_positions = [min(dataset_length, x + split_step)
                         for x in start_positions]

        return list(zip(start_positions, end_positions))
