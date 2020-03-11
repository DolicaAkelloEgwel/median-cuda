import time
from typing import List

import cupy as cp
import numpy as np
from cupy.cuda.memory import set_allocator

from imagingtester import (
    ImagingTester,
    PRINT_INFO,
    get_array_partition_indices,
    memory_needed_for_arrays,
    load_median_filter_file,
)
from imagingtester import num_partitions_needed as number_of_partitions_needed

LIB_NAME = "cupy"
MAX_CUPY_MEMORY = 0.8  # Anything exceeding this seems to make malloc fail for me

REFLECT_MODE = "reflect"

# Allocate CUDA memory
mempool = cp.get_default_memory_pool()
with cp.cuda.Device(0):
    mempool.set_limit(fraction=MAX_CUPY_MEMORY)
# mempool.malloc(mempool.get_limit())


def print_memory_metrics():
    """
    Print some information about how much space is being used on the GPU.
    """
    if not PRINT_INFO:
        return
    print("Used bytes:", mempool.used_bytes(), "/ Total bytes:", mempool.total_bytes())


def synchronise():
    cp.cuda.Stream.null.synchronize()
    cp.cuda.runtime.deviceSynchronize()


def get_synchronized_time():
    """
    Get the time after calling the cuda synchronize method. This should ensure the GPU has completed whatever it was
    doing before getting the time.
    """
    synchronise()
    return time.time()


def free_memory_pool(arrays=[]):
    """
    Delete the existing GPU arrays and free blocks so that successive calls to `_send_arrays_to_gpu` don't lead to any
    problems.
    """
    synchronise()
    if arrays:
        arrays.clear()
    synchronise()
    mempool.free_all_blocks()
    print_memory_metrics()


def _create_pinned_memory(cpu_array):
    """
    Use pinned memory as opposed to `asarray`. This allegedly this makes transferring quicker.
    :param cpu_array: The numpy array.
    :return: src
    """
    mem = cp.cuda.alloc_pinned_memory(cpu_array.nbytes)
    src = np.frombuffer(mem, cpu_array.dtype, cpu_array.size).reshape(cpu_array.shape)
    src[...] = cpu_array
    return src


def time_function(func):
    """
    Time an operation using a call to cupy's deviceSynchronize.
    :param func: The function to be timed.
    :return: The time the function took to complete its execution in seconds.
    """
    start = get_synchronized_time()
    func()
    return get_synchronized_time() - start


def get_free_bytes():
    free_bytes = mempool.free_bytes()
    if free_bytes > 0:
        return free_bytes
    return mempool.get_limit()


loaded_from_source = load_median_filter_file()

median_filter_module = cp.RawModule(code=loaded_from_source, backend="nvcc")
median_filter = median_filter_module.get_function("median_filter")


def cupy_median_filter(data, padded_data, filter_height, filter_width):
    N = 10
    block_size = (N, N, N)
    grid_size = (
        data.shape[0] // block_size[0],
        data.shape[1] // block_size[1],
        data.shape[2] // block_size[2],
    )
    median_filter(
        grid_size,
        block_size,
        (
            data,
            padded_data,
            data.shape[0],
            data.shape[1],
            data.shape[2],
            filter_height,
            filter_width,
        ),
    )


def replace_gpu_array_contents(gpu_array, cpu_array):
    gpu_array.set(cpu_array, cp.cuda.Stream(non_blocking=True))


class CupyImplementation(ImagingTester):
    def __init__(self, size, dtype, pinned_memory=False):
        super().__init__(size, dtype)

        # Determine how to pass data to the GPU based on the pinned_memory argument
        if pinned_memory:
            self._send_arrays_to_gpu = self._send_arrays_to_gpu_with_pinned_memory
        else:
            self._send_arrays_to_gpu = self._send_arrays_to_gpu_without_pinned_memory

        self.lib_name = LIB_NAME

    def _send_arrays_to_gpu_with_pinned_memory(self, cpu_arrays):
        """
        Transfer the arrays to the GPU using pinned memory. Should make data transfer quicker.
        """
        gpu_arrays = []

        if not isinstance(cpu_arrays, List):
            cpu_arrays = [cpu_arrays]

        for i in range(len(cpu_arrays)):
            try:
                pinned_memory = _create_pinned_memory(cpu_arrays[i].copy())
                gpu_array = cp.empty(pinned_memory.shape, dtype=self.dtype)
                array_stream = cp.cuda.Stream(non_blocking=True)
                gpu_array.set(pinned_memory, stream=array_stream)
                gpu_arrays.append(gpu_array)
            except cp.cuda.memory.OutOfMemoryError:
                print("Out of memory...")
                print_memory_metrics()
                self.print_memory_after_exception(cpu_arrays, gpu_arrays)
                return []

        if len(gpu_arrays) == 1:
            return gpu_arrays[0]
        return gpu_arrays

    def _send_arrays_to_gpu_without_pinned_memory(self, cpu_arrays):
        """
        Transfer the arrays to the GPU without using pinned memory.
        """
        gpu_arrays = []

        if not isinstance(cpu_arrays, List):
            cpu_arrays = [cpu_arrays]

        for cpu_array in cpu_arrays:
            try:
                gpu_array = cp.asarray(cpu_array.copy())
            except cp.cuda.memory.OutOfMemoryError:
                self.print_memory_after_exception(cpu_arrays, gpu_arrays)
                return []
            gpu_arrays.append(gpu_array)

        gpu_arrays = [cp.asarray(cpu_array) for cpu_array in cpu_arrays]

        if len(gpu_arrays) == 1:
            return gpu_arrays[0]
        return gpu_arrays

    def print_memory_after_exception(self, cpu_arrays, gpu_arrays):
        print(
            "Failed to make %s GPU arrays of size %s."
            % (len(cpu_arrays), cpu_arrays[0].shape)
        )
        print(
            "Used bytes:",
            mempool.used_bytes(),
            "/ Free bytes:",
            mempool.free_bytes(),
            "/ Space needed:",
            memory_needed_for_arrays(cpu_arrays),
        )
        free_memory_pool(gpu_arrays)

    def timed_median_filter(self, runs, filter_size):

        # Synchronize and free memory before making an assessment about available space
        free_memory_pool()

        # Determine the number of partitions required (not taking the padding into account)
        n_partitions_needed = number_of_partitions_needed(
            self.cpu_arrays[:1], get_free_bytes()
        )

        transfer_time = 0
        operation_time = 0

        pad_height = filter_size[1] // 2
        pad_width = filter_size[0] // 2

        filter_height = filter_size[0]
        filter_width = filter_size[1]

        padded_cpu_array = np.pad(
            self.cpu_arrays[0],
            pad_width=((0, 0), (pad_width, pad_width), (pad_height, pad_height)),
        )

        if n_partitions_needed == 1:

            # Time the transfer from CPU to GPU
            start = get_synchronized_time()
            gpu_data_array, padded_gpu_array = self._send_arrays_to_gpu(
                [self.cpu_arrays[0], padded_cpu_array]
            )
            transfer_time = get_synchronized_time() - start

            # Repeat the operation
            for _ in range(runs):
                operation_time += time_function(
                    lambda: cupy_median_filter(
                        gpu_data_array, padded_gpu_array, filter_height, filter_width
                    )
                )

            # Time the transfer from GPU to CPU
            transfer_time += time_function(gpu_data_array[0].get)

            # Free the GPU arrays
            free_memory_pool([gpu_data_array, padded_gpu_array])

        else:

            # Determine the number of partitions required again (to be on the safe side)
            n_partitions_needed = number_of_partitions_needed(
                [self.cpu_arrays[0], padded_cpu_array], get_free_bytes()
            )

            indices = get_array_partition_indices(
                self.cpu_arrays[0].shape[0], n_partitions_needed
            )

            start = get_synchronized_time()
            gpu_arrays = self._send_arrays_to_gpu(
                [
                    np.empty_like(
                        self.cpu_arrays[0][indices[0][0] : indices[0][1] :, :]
                    ),
                    np.empty_like(padded_cpu_array)[indices[0][0] : indices[0][1] :, :],
                ]
            )
            transfer_time += get_synchronized_time() - start

            # Return 0 when GPU is out of space
            if not gpu_arrays:
                return 0

            gpu_data_array, gpu_padded_array = gpu_arrays

            for i in range(n_partitions_needed):

                # Retrieve the segments used for this iteration of the operation
                split_cpu_array = self.cpu_arrays[0][indices[i][0] : indices[i][1] :, :]

                if split_cpu_array.shape == gpu_data_array.shape:
                    # Time transferring the segments to the GPU
                    start = get_synchronized_time()
                    gpu_data_array.set(
                        split_cpu_array, cp.cuda.Stream(non_blocking=True)
                    )
                    gpu_padded_array.set(
                        np.pad(
                            split_cpu_array,
                            pad_width=(
                                (0, 0),
                                (pad_width, pad_width),
                                (pad_height, pad_height),
                            ),
                            mode=REFLECT_MODE,
                        ),
                        cp.cuda.Stream(non_blocking=True),
                    )
                    transfer_time += get_synchronized_time() - start
                else:

                    diff = gpu_data_array.shape[0] - split_cpu_array.shape[0]

                    expanded_cpu_array = np.pad(
                        split_cpu_array, pad_width=((0, diff), (0, 0), (0, 0))
                    )
                    start = get_synchronized_time()
                    gpu_data_array.set(
                        expanded_cpu_array, cp.cuda.Stream(non_blocking=True)
                    )
                    transfer_time += get_synchronized_time() - start

                    padded_cpu_array = np.pad(
                        expanded_cpu_array,
                        pad_width=(
                            (0, 0),
                            (pad_width, pad_width),
                            (pad_height, pad_height),
                        ),
                    )
                    start = get_synchronized_time()
                    gpu_padded_array.set(
                        padded_cpu_array, cp.cuda.Stream(non_blocking=True)
                    )
                    transfer_time += get_synchronized_time() - start

                try:
                    # Carry out the operation on the slices
                    for _ in range(runs):
                        operation_time += time_function(
                            lambda: cupy_median_filter(
                                gpu_data_array,
                                gpu_padded_array,
                                filter_height,
                                filter_width,
                            )
                        )
                except cp.cuda.memory.OutOfMemoryError as e:
                    print(
                        "Unable to make extra arrays during operation despite successful transfer."
                    )
                    print(e)
                    free_memory_pool([gpu_data_array, gpu_padded_array])
                    return 0

                # Store time taken to transfer result
                transfer_time += time_function(gpu_data_array[0].get)

                # Free GPU arrays
                free_memory_pool([gpu_padded_array, gpu_data_array])

        self.print_operation_times(
            total_time=operation_time,
            operation_name="Median Filter",
            runs=runs,
            transfer_time=transfer_time,
        )

        return transfer_time + operation_time / runs
