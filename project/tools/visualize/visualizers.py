"""Module contains functions for volume visualization."""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def aggregation_visualize(
    volume: np.ndarray,
    aggregation: Callable = np.max,
    title: str = ''
):
    """
    Create grid with 3 images argregated from volume along corresponding axis.

    Parameters
    ----------
    volume : np.ndarray
        Array with volume to process.
    aggregation : Callable
        Function that aggregate data along specified axis.
    title: str
        Title for figure.
    """
    fig = plt.figure(figsize=(5 * 3, 5))
    fig.canvas.manager.set_window_title(title)
    c = 3
    for j in range(c):
        ax = fig.add_subplot(1, c, j + 1)
        ax.set_title(f'Axis {j + 1}')
        ax.imshow(aggregation(volume, axis=j), cmap='gray')
    fig.show()
    plt.show()
    plt.close()


def slice_visualize(
    volume: np.ndarray,
    title: str = ''
):
    """
    Visualize volume by showing its slices along axial axis.

    Parameters
    ----------
    volume : np.ndarray
        Array with volume to process.
    title : str
        Title for figure.
    """
    for i in range(volume.shape[-1]):
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(volume[..., i], cmap='gray')
        plt.title(f'{title} Slice #{i}')
        fig.show()
        plt.show()
        plt.close()
