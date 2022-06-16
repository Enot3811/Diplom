"""Script for testing tfrecord I/O tools."""

import argparse
import sys
import json
import csv
from pathlib import Path
from multiprocessing import cpu_count

import tensorflow as tf
from loguru import logger

PROJ_ROOT = Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(str(PROJ_ROOT))
from project.tools.profiling.dataset_profiling import (
    profile_dataset, do_initialization_iteration)
from project.tools.argparse.types import natural_int
from project.pipeline.tfrecord_reading.deserialization import (
    deserialize_example)
from project.pipeline.augmentations.augmentations import (
    tf_process_ct, tf_synth_augmentation)


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    in_gr = parser.add_argument_group('Inputs')
    in_gr.add_argument('--config',
                       help='Path to json config for test',
                       type=Path,
                       required=True)
    in_gr.add_argument('--num-iter',
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


def run_test(config_path: Path, num_iter: int = 1):
    """Run test."""
    with open(config_path, 'r') as read_file:
        test_config = json.load(read_file)

    results = []

    for i, test_case in enumerate(test_config['tests']):

        dir_path = Path(test_case['path'])
        disable_order = test_case['disable_order']
        crop_shape = tuple(test_case['crop_shape'])
        augmentation = test_case['augmentation']
        min_v = -1024
        max_v = 3072

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

        if i == 0:
            do_initialization_iteration(dset)

        title = '; '.join([f'{k} - {v}' for k, v in test_case.items()])
        title = f'Test {i + 1}'
        avg_time = profile_dataset(dset, num_iter, title)

        results.append({**test_case, 'avg_time': avg_time})
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
