"""
Script for testing TF I/O default tools.

It allows to vary
1) Amount of datasets to load from
1) Each dataset split
2) Size of loading batch
"""

import argparse
import csv
import json
import sys
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Dict, Any

import tensorflow as tf
from loguru import logger

PROJ_ROOT = Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(str(PROJ_ROOT))
from project.pipeline.h5_reading.split_manager import H5SplitManager
from project.pipeline.h5_reading.samplers import CTSampler, CTAugmentedSampler
from project.tools.profiling.dataset_profiling import (
    profile_dataset, do_initialization_iteration)
from project.tools.argparse.types import natural_int


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    in_gr = parser.add_argument_group('Inputs')
    in_gr.add_argument(
        '--config',
        help='Path to json config for test',
        type=Path,
        required=True)
    in_gr.add_argument(
        '--num-iter',
        help='Number of iterations per test.',
        type=natural_int,
        default=1)

    out_gr = parser.add_argument_group('Outputs')
    out_gr.add_argument(
        '--log-file',
        help='Path to file with logs',
        type=Path)
    out_gr.add_argument(
        '--out',
        help='Path to csv file with the results.',
        type=Path)

    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f'Config file "{args.config}" does not exist.')

    if args.out and args.out.exists():
        logger.warning(f'Out file "{args.out}" already exists and '
                       'will be overwritten. '
                       'Press enter to continue or stop running.')
        input()

    return args


def run_test(config_path: Path, num_iter: int = 1) -> List[Dict[str, Any]]:
    """
    Run I/O test.

    Iterate over dataset with the specified parameters and check average time.
    Test allows to vary:
      * datasets amount
      * dataset split
      * size of loading buffer

    Parameters
    ----------
    config_path : str
        Path to json test configuration file.
    num_iter : int, optional
        The number of iterations over dataset to calculate the average
        time, by default 1.

    Returns
    -------
    List[Dict[str, Any]]:
        List with results that consist of manager configuration & reading time.
    """
    results = []
    output_signature = (
        tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.string)
    )

    with open(config_path, 'r') as read_file:
        test_config = json.load(read_file)

    logger.debug('Create generators...')
    generator_configs = test_config['generators']
    generators = {}
    for g_name, g_params in generator_configs.items():
        g_class = g_params['class']
        del g_params['class']
        if g_class == 'CTSampler':
            generators[g_name] = CTSampler(**g_params)
        elif g_class == 'CTAugmentedSampler':
            generators[g_name] = CTAugmentedSampler(**g_params)
    logger.debug('OK.')

    logger.debug('Iterate through tests...')
    for i, test_case in enumerate(test_config['tests']):
        ds_list = test_case['datasets']
        manager = H5SplitManager(ds_list, generators)
        logger.debug(f'Amount of different managers is: {len(manager)}')

        # If there is only one dataset then there is no split
        # and then .interleave is not needed
        if len(manager) == 1:
            logger.debug('Create generator ds...')
            dset = tf.data.Dataset.from_generator(
                manager.__getitem__,
                output_signature=output_signature,
                args=(0,)
            ).unbatch().batch(16).prefetch(1)
        else:
            logger.debug('Create interleave ds...')
            dset = tf.data.Dataset.range(len(manager)).interleave(
                lambda indx: tf.data.Dataset.from_generator(
                    manager.__getitem__,
                    output_signature=output_signature,
                    args=(indx,)
                ),
                num_parallel_calls=cpu_count()
            ).unbatch().batch(16).prefetch(1)

        params = {}
        for j, ds in enumerate(ds_list):
            for k, v in ds.items():
                params[f'{j + 1}_{k}'] = v
        title = '; '.join([f'{k} - {v}' for k, v in params.items()])
        if i == 0:
            do_initialization_iteration(dset)
        avg_time = profile_dataset(dset, title=title, num_iter=num_iter)
        results.append({**params, 'avg_time': avg_time})
    logger.debug('OK.')

    return results


def main():
    """Application entry point."""
    args = parse_args()

    if args.log_file:
        Path(args.log_file.parent).mkdir(parents=True, exist_ok=True)
        logger.add(args.log_file, format='{message}', level='DEBUG')

    results = run_test(args.config, args.num_iter)

    logger.info(f'Results are:\n{results}\n')

    if args.out:
        Path(args.out.parent).mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile,
                                    fieldnames=list(results[0].keys()),
                                    delimiter=';')
            writer.writeheader()
            for result in results:
                writer.writerow(result)


if __name__ == '__main__':
    main()
