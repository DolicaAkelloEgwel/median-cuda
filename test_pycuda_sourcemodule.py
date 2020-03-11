from pycuda import gpuarray
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import numpy as np

from imagingtester import (
    DTYPE,
    create_arrays,
    SIZES_SUBSET,
    num_partitions_needed,
    N_RUNS,
    get_array_partition_indices,
    FILTER_SIZE,
    FREE_MEMORY_FACTOR,
)
from cpu_imaging_filters import scipy_median_filter
from pycuda_test_utils import (
    PyCudaImplementation,
    get_free_bytes,
    get_time,
    time_function,
    free_memory_pool,
    get_median_filter_string,
    get_total_bytes,
    get_used_bytes,
)

from write_and_read_results import ARRAY_SIZES, write_results_to_file

LIB_NAME = "pycuda"
mode = "sourcemodule"
REFLECT_MODE = "reflect"

median_filter_module = SourceModule(get_median_filter_string())
median_filter = median_filter_module.get_function("median_filter")


def pycuda_median_filter(data, padded_data, filter_height, filter_width):
    median_filter(
        data,
        padded_data,
        np.int32(data.shape[0]),
        np.int32(data.shape[1]),
        np.int32(data.shape[2]),
        np.int32(filter_height),
        np.int32(filter_width),
        block=(10, 10, 10),
    )


class PyCudaSourceModuleImplementation(PyCudaImplementation):
    def __init__(self, size, dtype):

        super().__init__(size, dtype)
        self.warm_up()

    def warm_up(self):
        warm_up_size = (2, 2, 2)
        cpu_data_array = create_arrays(warm_up_size, DTYPE)[0]

        filter_height = 3
        filter_width = 3
        pad_height = filter_height // 2
        pad_width = filter_width // 2

        padded_cpu_array = np.pad(
            cpu_data_array,
            pad_width=((0, 0), (pad_height, pad_height), (pad_width, pad_width)),
            mode=REFLECT_MODE,
        )
        gpu_data_array, gpu_padded_array = self._send_arrays_to_gpu(
            [cpu_data_array, padded_cpu_array]
        )
        pycuda_median_filter(
            gpu_data_array, gpu_padded_array, filter_height, filter_width
        )

    def timed_median_filter(self, runs, filter_size):

        n_partitions_needed = num_partitions_needed(
            self.cpu_arrays[:1], get_free_bytes()
        )

        transfer_time = 0
        operation_time = 0

        pad_height = filter_size[1] // 2
        pad_width = filter_size[0] // 2

        filter_height = filter_size[0]
        filter_width = filter_size[1]

        if n_partitions_needed == 1:

            # Time transfer from CPU to GPU (and padding creation)
            start = get_time()
            cpu_padded_array = np.pad(
                self.cpu_arrays[0],
                pad_width=((0, 0), (pad_width, pad_width), (pad_height, pad_height)),
                mode=REFLECT_MODE,
            )
            gpu_data_array, gpu_padded_array = self._send_arrays_to_gpu(
                [self.cpu_arrays[0], cpu_padded_array]
            )
            transfer_time += get_time() - start

            # Repeat the operation
            for _ in range(runs):
                operation_time += time_function(
                    lambda: pycuda_median_filter(
                        gpu_data_array, gpu_padded_array, filter_height, filter_width
                    )
                )

            # Time the transfer from GPU to CPU
            transfer_time += time_function(lambda: gpu_data_array.get_async)

            # Free the GPU arrays
            free_memory_pool([gpu_data_array, gpu_padded_array])

        else:

            n_partitions_needed = num_partitions_needed(
                self.cpu_arrays[:1], get_free_bytes()
            )

            indices = get_array_partition_indices(
                self.cpu_arrays[0].shape[0], n_partitions_needed
            )

            for i in range(n_partitions_needed):

                # Retrieve the segments used for this iteration of the operation
                split_cpu_array = self.cpu_arrays[0][indices[i][0] : indices[i][1] :, :]

                # Time transferring the segments to the GPU
                cpu_padded_array = np.pad(
                    split_cpu_array,
                    pad_width=(
                        (0, 0),
                        (pad_width, pad_width),
                        (pad_height, pad_height),
                    ),
                    mode=REFLECT_MODE,
                )
                start = get_time()
                gpu_data_array, gpu_padded_array = self._send_arrays_to_gpu(
                    [split_cpu_array, cpu_padded_array]
                )
                transfer_time += get_time() - start

                # Carry out the operation on the slices
                for _ in range(runs):
                    operation_time += time_function(
                        lambda: pycuda_median_filter(
                            gpu_data_array,
                            gpu_padded_array,
                            filter_height,
                            filter_width,
                        )
                    )

                transfer_time += time_function(lambda: gpu_data_array.get_async)

                # Free the GPU arrays
                free_memory_pool([gpu_data_array, gpu_padded_array])

        if transfer_time > 0 and operation_time > 0:
            self.print_operation_times(
                operation_time, "median filter", runs, transfer_time
            )

        self.synchronise()

        return transfer_time + operation_time / N_RUNS


# These need to be odd values and need to be equal
filter_height = 3
filter_width = 3
filter_size = (filter_height, filter_width)

np_data = np.random.uniform(low=0.0, high=10.0, size=(3, 3, 3)).astype(DTYPE)

# Create a padded array in the GPU
pad_height = filter_height // 2
pad_width = filter_width // 2
padded_data = np.pad(
    np_data,
    pad_width=((0, 0), (pad_height, pad_height), (pad_width, pad_width)),
    mode=REFLECT_MODE,
)

gpu_data = gpuarray.GPUArray(shape=np_data.shape, dtype=np_data.dtype)
gpu_data.set_async(np_data)
gpu_padded_data = gpuarray.GPUArray(shape=padded_data.shape, dtype=padded_data.dtype)
gpu_padded_data.set_async(padded_data)

# Run the median filter on the GPU
print(gpu_data[0])
pycuda_median_filter(gpu_data, gpu_padded_data, filter_height, filter_width)
print(gpu_data[0])
# Run the scipy median filter
scipy_median_filter(np_data, size=filter_size)
# Check that the results match
print(np_data[0])

print(np.isclose(np_data[0], gpu_data[0].get()))
assert np.allclose(np_data, gpu_data.get())

median_filter_results = []


free_memory_pool([gpu_data, gpu_padded_data])
print("Free bytes", get_free_bytes())


for size in ARRAY_SIZES[:SIZES_SUBSET]:

    obj = PyCudaSourceModuleImplementation(size, DTYPE)
    avg_median = obj.timed_median_filter(N_RUNS, FILTER_SIZE)
    print("Free bytes after run:", get_free_bytes())

    if avg_median > 0:
        median_filter_results.append(avg_median)

    write_results_to_file([LIB_NAME, mode], "median filter", median_filter_results)
