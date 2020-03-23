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
    FREE_MEMORY_FACTOR,
    create_arrays,
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
        return free_bytes * FREE_MEMORY_FACTOR
    return mempool.get_limit() * FREE_MEMORY_FACTOR


loaded_from_source = load_median_filter_file()

median_filter_module = cp.RawModule(code=loaded_from_source, backend="nvcc")
three_dim_median_filter = median_filter_module.get_function("three_dim_median_filter")
two_dim_median_filter = median_filter_module.get_function("two_dim_median_filter")
two_dim_remove_light_outliers = median_filter_module.get_function(
    "two_dim_remove_light_outliers"
)
two_dim_remove_dark_outliers = median_filter_module.get_function(
    "two_dim_remove_dark_outliers"
)


def cupy_three_dim_median_filter(data, padded_data, filter_size):
    N = 10
    block_size = (N, N, N)
    grid_size = (
        data.shape[0] // block_size[0],
        data.shape[1] // block_size[1],
        data.shape[2] // block_size[2],
    )
    three_dim_median_filter(
        grid_size,
        block_size,
        (data, padded_data, data.shape[0], data.shape[1], data.shape[2], filter_size),
    )


def cupy_two_dim_median_filter(data, padded_data, filter_size):
    block_size, grid_size = create_block_and_grid_args(data)
    print("Data shape:", data.shape)
    print("Grid size:", grid_size)
    print("Block size:", block_size)
    two_dim_median_filter(
        grid_size,
        block_size,
        (data, padded_data, data.shape[0], data.shape[1], filter_size),
    )


def create_block_and_grid_args(data):
    N = 10
    block_size = (N, N)
    if data.shape[0] > 10 and data.shape[1] > 10:
        grid_size = (
            data.shape[0] // block_size[0],
            data.shape[1] // block_size[1],
            data.shape[2] // block_size[1],
        )
    else:
        grid_size = (data.shape[0], data.shape[1], data.shape[1])
    return block_size, grid_size


def cupy_two_dim_remove_outliers(data, padded_data, diff, size, mode):
    block_size, grid_size = create_block_and_grid_args(10, data)

    if mode == "light":
        two_dim_remove_light_outliers(
            grid_size,
            block_size,
            (data, padded_data, data.shape[0], data.shape[1], size, diff),
        )
    if mode == "dark":
        two_dim_remove_dark_outliers(
            grid_size,
            block_size,
            (data, padded_data, data.shape[0], data.shape[1], size, diff),
        )


def create_padded_array(arr, pad_size, mode="reflect"):

    if arr.ndim == 2:
        return np.pad(
            arr, pad_width=((pad_size, pad_size), (pad_size, pad_size)), mode=mode
        )
    else:
        return np.pad(
            arr,
            pad_width=((0, 0), (pad_size, pad_size), (pad_size, pad_size)),
            mode=mode,
        )


def replace_gpu_array_contents(
    gpu_array, cpu_array, stream=cp.cuda.Stream(non_blocking=True)
):
    gpu_array.set(cpu_array, stream)


class CupyImplementation(ImagingTester):
    def __init__(self, size, dtype, pinned_memory=False):
        super().__init__(size, dtype)

        # Determine how to pass data to the GPU based on the pinned_memory argument
        if pinned_memory:
            self._send_arrays_to_gpu = self._send_arrays_to_gpu_with_pinned_memory
        else:
            self._send_arrays_to_gpu = self._send_arrays_to_gpu_without_pinned_memory

        self.lib_name = LIB_NAME
        self._warm_up()

    def _warm_up(self):

        image_stack = create_arrays((3, 3, 3), self.dtype)[0]
        filter_size = 3

        single_image = image_stack[0]
        padded_image = create_padded_array(single_image, filter_size)
        gpu_image, gpu_padded_image = self._send_arrays_to_gpu_without_pinned_memory(
            [single_image, padded_image]
        )

        cupy_two_dim_median_filter(gpu_image, gpu_padded_image, filter_size)

        padded_image_stack = create_padded_array(image_stack, filter_size // 2)
        gpu_image_stack, gpu_padded_image_stack = self._send_arrays_to_gpu_without_pinned_memory(
            [image_stack, padded_image_stack]
        )

        cupy_three_dim_median_filter(
            gpu_image_stack, gpu_padded_image_stack, filter_size
        )

    def _send_arrays_to_gpu_with_pinned_memory(self, cpu_arrays, streams=None):
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
                if streams is None:
                    gpu_array.set(
                        pinned_memory, stream=cp.cuda.Stream(non_blocking=True)
                    )
                else:
                    gpu_array.set(pinned_memory, stream=streams[i])
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

    def timed_async_median_filter(self, runs, filter_size):

        # Synchronize and free memory before making an assessment about available space
        free_memory_pool()

        operation_time = 0
        transfer_time = 0

        pad_size = self.get_padding_value(filter_size)

        n_images = self.cpu_arrays[0].shape[0]

        MAX_GPU_SLICES = 180

        if n_images > MAX_GPU_SLICES:
            slice_limit = MAX_GPU_SLICES
        else:
            slice_limit = n_images

        cpu_data_slices = [
            self.cpu_arrays[0][i] for i in range(self.cpu_arrays[0].shape[0])
        ]
        cpu_padded_slices = [
            create_padded_array(arr, pad_size) for arr in cpu_data_slices
        ]
        streams = [cp.cuda.Stream(non_blocking=True) for _ in range(slice_limit)]

        start = get_synchronized_time()
        gpu_data_slices = self._send_arrays_to_gpu_with_pinned_memory(
            cpu_data_slices[:slice_limit], streams
        )
        gpu_padded_data = self._send_arrays_to_gpu_with_pinned_memory(
            cpu_padded_slices[:slice_limit], streams
        )
        transfer_time += get_synchronized_time() - start

        print("Copied arrays.")

        start = get_synchronized_time()
        for i in range(self.cpu_arrays[0].shape[0]):

            streams[i % slice_limit].synchronize()

            replace_gpu_array_contents(
                gpu_data_slices[i % slice_limit],
                self.cpu_arrays[0][i],
                streams[i % slice_limit],
            )

            replace_gpu_array_contents(
                gpu_padded_data[i % slice_limit],
                cpu_padded_slices[i],
                streams[i % slice_limit],
            )

            streams[i % slice_limit].synchronize()

            cupy_two_dim_median_filter(
                gpu_data_slices[i % slice_limit],
                gpu_padded_data[i % slice_limit],
                filter_size,
            )

            streams[i % slice_limit].synchronize()

            self.cpu_arrays[0][i][:] = gpu_data_slices[i % slice_limit].get(
                streams[i % slice_limit]
            )
        operation_time += get_synchronized_time() - start

        free_memory_pool(gpu_data_slices + gpu_padded_data)

        return operation_time + transfer_time

    def timed_three_dim_median_filter(self, runs, filter_size):

        # Synchronize and free memory before making an assessment about available space
        free_memory_pool()

        # Determine the number of partitions required (not taking the padding into account)
        n_partitions_needed = number_of_partitions_needed(
            self.cpu_arrays[:1], get_free_bytes()
        )

        transfer_time = 0
        operation_time = 0

        pad_size = self.get_padding_value(filter_size)

        padded_cpu_array = create_padded_array(self.cpu_arrays[0], pad_size)

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
                    lambda: cupy_three_dim_median_filter(
                        gpu_data_array, padded_gpu_array, filter_size
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
                        create_padded_array(split_cpu_array, pad_size),
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

                    padded_cpu_array = create_padded_array(expanded_cpu_array, pad_size)
                    start = get_synchronized_time()
                    gpu_padded_array.set(
                        padded_cpu_array, cp.cuda.Stream(non_blocking=True)
                    )
                    transfer_time += get_synchronized_time() - start

                try:
                    # Carry out the operation on the slices
                    for _ in range(runs):
                        operation_time += time_function(
                            lambda: cupy_three_dim_median_filter(
                                gpu_data_array, gpu_padded_array, filter_size
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

    def get_padding_value(self, filter_size):
        return filter_size // 2
