import logging

import numpy as np

import cutde.cpp_backend_double
import cutde.cpp_backend_float

from .gpu_backend import clear_all_caches as _clear_backend_caches

logger = logging.getLogger(__name__)


def clear_gpu_memory():
    """
    Clear cached modules and force garbage collection to free memory.

    For the C++ backend, this primarily clears the module cache. While the C++
    backend doesn't use GPU memory, this function is provided for API
    consistency with the CUDA and OpenCL backends.
    """
    import gc

    _clear_backend_caches()
    gc.collect()
    logger.debug("Cleared caches and triggered garbage collection")


def to(arr, float_type):
    return arr.ravel().astype(float_type)


def zeros(shape, float_type):
    return np.zeros(shape, dtype=float_type)


def empty(shape, float_type):
    return np.empty(shape, dtype=float_type)


def get(arr):
    return arr


def max_block_size(requested):
    return 1


def load_module(
    tmpl_name, tmpl_dir=None, save_code=False, no_caching=False, tmpl_args=None
):
    if tmpl_args["float_type"] == "float":
        return cutde.cpp_backend_float
    else:
        return cutde.cpp_backend_double
