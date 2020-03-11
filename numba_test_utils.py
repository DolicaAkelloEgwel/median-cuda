from numba import cuda, vectorize

from imagingtester import (
    ImagingTester,
    PRINT_INFO,
    DTYPE,
    MINIMUM_PIXEL_VALUE,
    memory_needed_for_arrays,
)

LIB_NAME = "numba"


def get_free_bytes():
    return cuda.current_context().get_memory_info()[0]


def get_total_bytes():
    return cuda.current_context().get_memory_info()[1]


def get_used_bytes():
    return get_total_bytes() - get_free_bytes()


def create_vectorise_add_arrays(target):
    @vectorize(["{0}({0},{0})".format(DTYPE)], target=target)
    def add_arrays(elem1, elem2):
        return elem1 + elem2

    return add_arrays


def create_vectorise_background_correction(target):
    @vectorize("{0}({0},{0},{0},{0},{0})".format(DTYPE), target=target)
    def background_correction(data, dark, flat, clip_min, clip_max):

        flat -= dark

        if flat == 0:
            flat = MINIMUM_PIXEL_VALUE

        data -= dark
        data /= flat

        if data < clip_min:
            data = clip_min
        if data > clip_max:
            data = clip_max

        return data

    return background_correction


def print_memory_info_after_transfer_failure(arr, n_gpu_arrays_needed):
    print("Failed to make %s GPU arrays of size %s." % (n_gpu_arrays_needed, arr.shape))
    print(
        "Used bytes:",
        get_used_bytes(),
        "/ Total bytes:",
        get_total_bytes(),
        "/ Space needed:",
        memory_needed_for_arrays(arr, n_gpu_arrays_needed),
    )


class NumbaImplementation(ImagingTester):
    def __init__(self, size, dtype):
        super().__init__(size, dtype)
        self.warm_up()
        self.lib_name = LIB_NAME
        self.streams = []

    def warm_up(self):
        pass

    def get_time(self):
        pass

    def time_function(self, func):
        start = self.get_time()
        func()
        return self.get_time() - start

    def synchronise(self):

        for stream in self.streams:
            stream.synchronize()
        cuda.synchronize()

    def clear_cuda_memory(self, gpu_arrays=[]):

        self.synchronise()

        if PRINT_INFO:
            print("Free bytes before clearing memory:", get_free_bytes())

        if gpu_arrays:
            for array in gpu_arrays:
                del array
                array = None
            del gpu_arrays

        cuda.current_context().deallocations.clear()
        self.synchronise()

        if PRINT_INFO:
            print("Free bytes after clearing memory:", get_free_bytes())

    def _send_arrays_to_gpu(self, arrays_to_transfer, n_gpu_arrays_needed):

        gpu_arrays = []
        stream = cuda.stream()
        self.streams.append(stream)
        with cuda.pinned(*arrays_to_transfer):
            for arr in arrays_to_transfer:
                try:
                    gpu_array = cuda.to_device(arr, stream)
                except cuda.cudadrv.driver.CudaAPIError:
                    print_memory_info_after_transfer_failure(arr, n_gpu_arrays_needed)
                    self.clear_cuda_memory(gpu_arrays)
                    return []
                gpu_arrays.append(gpu_array)
        return gpu_arrays

    def timed_imaging_operation(
        self, runs, alg, alg_name, n_arrs_needed, n_gpu_arrs_needed
    ):
        pass
