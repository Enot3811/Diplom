"""Module provides tools for tf profiling."""

import time

import numpy as np
import tensorflow as tf
from loguru import logger


def profile_dataset(dset: tf.data.Dataset, num_iter: int = 1,
                    title: str = '', delay: float = 0.1) -> float:
    """Iterate over dataset and check time.

    Keeps truck of iteration time over dataset
    and average iterating time over num_iter iterations.

    Parameters
    ----------
    dset : tf.data.Dataset
        Dataset over which to iterate
    num_iter : int, optional
        Number of iterations for calculation average time, by default 1
    title : str, optional
        How to title printed average iteration time, by default ''
    delay : float, optional
        How many seconds to sleep to simulate computations per batch.

    Returns
    -------
    float
        Average time of iteration over dataset.
    """
    if num_iter < 1:
        raise ValueError(f'Wrong iterations for profile = "{num_iter}".')
    if title != '':
        title += ': '
    logger.debug(f'Start profiling TF-DS; Iterations = "{num_iter}".')
    times = []
    for i in range(num_iter):
        iter_time = time.perf_counter()
        for _ in dset:
            # Perform training step simulation
            time.sleep(delay)
        iter_time = time.perf_counter() - iter_time
        times.append(iter_time)
        logger.debug('{} iteration - {:.3f}'.format(i + 1, iter_time))

    avg_time = float(np.mean(times))
    logger.debug(title + 'Average time = {:.3f}'.format(avg_time))
    return avg_time


def do_initialization_iteration(dset: tf.data.Dataset) -> float:
    """
    Iterate over for initialize TF computitional graph.

    First iteration over dataset takes longer beacause TF needs to initialize
    its graph before doing calculations.
    So then additional iteration needed to get correct results.

    Parameters
    ----------
    dset : tf.data.Dataset
        Dataset over which to iterate

    Returns
    -------
    float
        Time of additional iteration.
    """
    logger.debug('Do first iteration for initialization TF graph...')
    iter_time = time.perf_counter()
    for _ in dset:
        pass
    iter_time = time.perf_counter() - iter_time
    logger.debug('Initialization iteration took {:.3f}.'.format(iter_time))

    return iter_time
