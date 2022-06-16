"""
Module containing deserialization functions for TFRecord datasets.
"""


from typing import Any, Dict, Optional, Tuple
from multiprocessing import cpu_count

import tensorflow as tf


def deserialize_example(
    record_bytes: tf.Tensor,
    record_shema: Optional[Dict[str, Any]] = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Decode one serialized example.

    Parameters
    ----------
    record_bytes : tf.Tensor
        Tensor contained bytes string

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor]
        Tensor with volume and tensor with name
    """
    if record_shema is None:
        record_shema = {
            'volume': tf.io.FixedLenFeature([], dtype=tf.string),
            'name': tf.io.FixedLenFeature([], dtype=tf.string)}

    parsed_example = tf.io.parse_single_example(
        # Data
        record_bytes,
        # Schema
        record_shema
    )

    parsed_example['volume'] = tf.io.parse_tensor(
        parsed_example['volume'], tf.float32)

    return (parsed_example['volume'], parsed_example['name'])


def deserialize_example_batch(
    record_bytes: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Parse batch of serialized examples.

    Parameters
    ----------
    record_bytes : tf.Tensor
        Tensor contained several bytes strings

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor]
        Tensor with volumes and tensor with names
    """
    parsed_batch = tf.io.parse_sequence_example(
        record_bytes,
        {'volume': tf.io.FixedLenFeature([], dtype=tf.string),
         'name': tf.io.FixedLenFeature([], dtype=tf.string)}
    )[0]
    parsed_batch['volume'] = tf.map_fn(
        lambda x: tf.io.parse_tensor(x, tf.float32),
        parsed_batch['volume'],
        fn_output_signature=tf.TensorSpec(shape=(None, None, None),
                                          dtype=tf.float32),
        parallel_iterations=cpu_count()
    )
    
    return (parsed_batch['volume'], parsed_batch['name'])
