"""Module with unit tests for pipelines."""


import sys
import json
from multiprocessing import cpu_count
from pathlib import Path

import tensorflow as tf
import h5py

PROJ_ROOT = Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(str(PROJ_ROOT))
from project.pipeline.h5_reading.split_manager import H5SplitManager
from project.pipeline.h5_reading.samplers import CTSampler, CTAugmentedSampler
from project.pipeline.tfrecord_reading.deserialization import (
    deserialize_example)
from project.pipeline.augmentations.augmentations import (
    tf_process_ct, tf_synth_augmentation)


def test_h5_io():
    """Test for h5 pipeline without augmentation."""
    config_paths = [
        # 'pipeline/tests/configs/test_h5_io_1.json',
        'pipeline/tests/configs/test_h5_io_2.json']

    output_signature = (
        tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.string)
    )

    for config_path in config_paths:

        with open(config_path, 'r') as read_file:
            test_config = json.load(read_file)

        generator_configs = test_config['generators']
        generators = {}
        for g_name, g_params in generator_configs.items():
            g_class = g_params['class']
            del g_params['class']
            if g_class == 'CTSampler':
                generators[g_name] = CTSampler(**g_params)
            else:
                raise ValueError('Specified generator ')

        for test_case in test_config['tests']:
            ds_list = test_case['datasets']
            manager = H5SplitManager(ds_list, generators)

            size = 0
            for ds in ds_list:
                with h5py.File(ds['h5_path']) as file:
                    size += file.attrs['size']

            # If there is only one dataset then there is no split
            # and then .interleave is not needed
            if len(manager) == 1:
                dset = tf.data.Dataset.from_generator(
                    manager.__getitem__,
                    output_signature=output_signature,
                    args=(0,)
                ).unbatch().batch(16).prefetch(1)
            else:
                dset = tf.data.Dataset.range(len(manager)).interleave(
                    lambda indx: tf.data.Dataset.from_generator(
                        manager.__getitem__,
                        output_signature=output_signature,
                        args=(indx,)
                    ),
                    num_parallel_calls=cpu_count()
                ).unbatch().batch(16).prefetch(1)
            
            samples = []
            for sample in dset:
                names = sample[1].numpy()
                for name in names:
                    samples.append(name)
            assert len(samples) == size
            assert len(set(samples)) == size


def test_h5_io_aug():
    """Test for h5 pipeline with augmentation."""
    config_paths = [
        'pipeline/tests/configs/test_aug_h5_io.json']

    output_signature = (
        tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.string)
    )

    for config_path in config_paths:
        with open(config_path, 'r') as read_file:
            test_config = json.load(read_file)

        for test_case in test_config['tests']:

            ds_list = test_case['datasets']

            size = 0
            for ds in ds_list:
                with h5py.File(ds['h5_path']) as file:
                    size += file.attrs['size']

            assert len(ds_list) == 1
            ds = ds_list[0]
            params = {k: v for k, v in ds.items()}

            # Augmentation in interleave variant
            generators = {
                'generator': CTAugmentedSampler(params['crop_shape'],
                                                params['buffer_size'])
            }
            ds['generator_name'] = 'generator'
            manager = H5SplitManager([ds], generators)

            # If there is only one dataset then there is no split
            # and then .interleave is not needed
            if len(manager) == 1:
                dset = tf.data.Dataset.from_generator(
                    manager.__getitem__,
                    output_signature=output_signature,
                    args=(0,)
                )
            else:
                dset = tf.data.Dataset.range(len(manager)).interleave(
                    lambda indx: tf.data.Dataset.from_generator(
                        manager.__getitem__,
                        output_signature=output_signature,
                        args=(indx,)
                    ),
                    num_parallel_calls=cpu_count()
                )
            dset = dset.unbatch().batch(16).prefetch(1)
            
            samples = []
            for sample in dset:
                names = sample[1].numpy()
                for name in names:
                    samples.append(name)
            assert len(samples) == size
            assert len(set(samples)) == size

            # Augmentation in the map variant
            generators = {
                'generator': CTSampler(params['crop_shape'],
                                       params['buffer_size'])
            }
            ds['generator_name'] = 'generator'
            manager = H5SplitManager([ds], generators)

            # If there is only one dataset then there is no split
            # and then .interleave is not needed
            if len(manager) == 1:
                dset = tf.data.Dataset.from_generator(
                    manager.__getitem__,
                    output_signature=output_signature,
                    args=(0,)
                )
                dset = dset.unbatch().batch(ds['buffer_size'])
            else:
                dset = tf.data.Dataset.range(len(manager)).interleave(
                    lambda indx: tf.data.Dataset.from_generator(
                        manager.__getitem__,
                        output_signature=output_signature,
                        args=(indx,)
                    ),
                    num_parallel_calls=cpu_count()
                )
            dset = (
                dset.map(tf_synth_augmentation, num_parallel_calls=cpu_count())
                    .unbatch().batch(16).prefetch(1))
            
            samples = []
            for sample in dset:
                names = sample[1].numpy()
                for name in names:
                    samples.append(name)
            assert len(samples) == size
            assert len(set(samples)) == size


def test_tfrecord_io():
    config_paths = [
        'pipeline/tests/configs/test_tfrecord_io.json'
    ]
    
    for config_path in config_paths:

        with open(config_path, 'r') as read_file:
            test_config = json.load(read_file)

        for i, test_case in enumerate(test_config['tests']):

            dir_path = Path(test_case['path'])
            disable_order = test_case['disable_order']
            crop_shape = tuple(test_case['crop_shape'])
            augmentation = test_case['augmentation']
            min_v = -1024
            max_v = 3072

            size = int(str(dir_path.name).split('_')[0][5:])

            file_paths = [str(file.absolute()) for file in dir_path.glob('*')]

            paths_dset = tf.data.Dataset.from_tensor_slices(file_paths)
            raw_dataset = tf.data.TFRecordDataset(paths_dset)

            if disable_order:
                # disable order, increase speed
                ignore_order = tf.data.Options()
                ignore_order.experimental_deterministic = False
                raw_dataset = raw_dataset.with_options(ignore_order)

            dset = (raw_dataset.map(deserialize_example, tf.data.AUTOTUNE)
                    .map(lambda ct, name: tf_process_ct(
                        ct, crop_shape, min_v, max_v, name), tf.data.AUTOTUNE)
                    .batch(16))

            if augmentation:
                dset = dset.map(tf_synth_augmentation,
                                num_parallel_calls=cpu_count())

            samples = []
            for sample in dset:
                names = sample[1].numpy()
                for name in names:
                    samples.append(name)
            assert len(samples) == size
            assert len(set(samples)) == size
