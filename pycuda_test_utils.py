import time

from imagingtester import (
    ImagingTester,
    memory_needed_for_arrays,
    DTYPE,
    FREE_MEMORY_FACTOR,
)
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit

if DTYPE == "float32":
    C_DTYPE = "float"
else:
    C_DTYPE = "double"

LIB_NAME = "pycuda"


def get_time():
    drv.Context.synchronize()
    return time.time()


def free_memory_pool(gpu_arrays):
    for gpu_array in gpu_arrays:
        gpu_array.gpudata.free()


def time_function(func):
    start = get_time()
    func()
    end = get_time()
    return end - start


def timed_get_array_from_gpu(gpu_array):
    start = get_time()
    gpu_array.get()
    end = get_time()
    return end - start


def get_free_bytes():
    return drv.mem_get_info()[0]


def get_total_bytes():
    return drv.mem_get_info()[1]


def get_used_bytes():
    return get_total_bytes() - get_free_bytes()


def print_memory_info_after_transfer_failure(cpu_arrays):
    print("Failed to make GPU arrays of size %s." % (str(cpu_arrays[0].shape)))
    print(
        "Used bytes:",
        get_used_bytes(),
        "/ Free bytes:",
        get_free_bytes(),
        "/ Space needed:",
        memory_needed_for_arrays(cpu_arrays),
        "/ Difference:",
        memory_needed_for_arrays(cpu_arrays) - get_free_bytes(),
    )


median_filter = """
__device__ float find_median(float* neighb_array, const int N)
{
    int i, j;
    float key;
    for (i = 1; i < N; i++)
    {
        key = neighb_array[i];
        j = i - 1;
        while (j >= 0 && neighb_array[j] > key)
        {
            neighb_array[j + 1] = neighb_array[j];
            j = j - 1;
        }
        neighb_array[j + 1] = key;
    }
    return neighb_array[N / 2];
}
__global__ void median_filter(float* data_array, const float* padded_array, const int N_IMAGES, const int X, const int Y, const int filter_height, const int filter_width)
{
    unsigned int id_img = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int id_x = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int id_y = blockIdx.z*blockDim.z + threadIdx.z;
    unsigned int n_counter = 0;
    unsigned int img_size =  X * Y;
    unsigned int padded_img_width =  X + filter_height - 1;
    unsigned int padded_img_size =  padded_img_width * (Y + filter_width - 1);

    float neighb_array[25];
    if ((id_img < N_IMAGES) && (id_x < X) && (id_y < Y))
    {
        for (int i = id_x; i < id_x + filter_height; i++)
        {
            for (int j = id_y; j < id_y + filter_width; j++)
            {
                neighb_array[n_counter] = padded_array[(id_img * padded_img_size) + (i * padded_img_width) + j];
                n_counter += 1;
            }
        }
        data_array[(id_img * img_size) + (id_x * X) + id_y] = find_median(neighb_array, filter_height * filter_width);
    }
}
"""


def get_median_filter_string():
    return median_filter.replace("float", C_DTYPE)


class PyCudaImplementation(ImagingTester):
    def __init__(self, size, dtype):
        super().__init__(size, dtype)
        self.lib_name = LIB_NAME
        self.streams = []

    def _send_arrays_to_gpu(self, cpu_arrays):

        gpu_arrays = []

        for cpu_array in cpu_arrays:
            try:
                gpu_array = gpuarray.GPUArray(
                    shape=cpu_array.shape, dtype=cpu_array.dtype
                )
            except drv.MemoryError as e:
                print_memory_info_after_transfer_failure(cpu_arrays)
                free_memory_pool(gpu_arrays)
                print(e)
                return []
            stream = drv.Stream()
            self.streams.append(stream)
            gpu_array.set_async(cpu_array, stream)
            gpu_arrays.append(gpu_array)

        return gpu_arrays

    def synchronise(self):
        for stream in self.streams:
            stream.synchronize()
        drv.Context.synchronize()
