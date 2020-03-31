import tomopy
import time
import scipy.ndimage as scipy_ndimage
from imagingtester import ImagingTester, N_RUNS, FILTER_SIZE


def scipy_median_filter(data, size):
    if data.ndim == 3:
        for idx in range(0, data.shape[0]):
            data[idx] = scipy_ndimage.median_filter(
                data[idx], (size, size), mode="mirror"
            )
    else:
        data = scipy_ndimage.median_filter(data, (size, size), mode="mirror")


def tomopy_remove_outlier(data, size, diff):
    data = tomopy.misc.corr.remove_outlier(data, diff, size)


def time_function(func):
    start = time.time()
    func()
    return time.time() - start


LIB_NAME = "cpu"


class CPUImplementation(ImagingTester):
    def __init__(self, size, dtype):
        super().__init__(size, dtype)
        self.lib_name = LIB_NAME

    def timed_median_filter(self, reps):
        data = self.cpu_arrays[0]
        total_time = 0
        for _ in range(reps):
            total_time += time_function(lambda: scipy_median_filter(data, FILTER_SIZE))
        operation_time = total_time / reps
        self.print_operation_times(
            total_time=total_time, operation_name="median filter", runs=reps
        )
        return operation_time

    def timed_remove_outlier(self, reps):
        data = self.cpu_arrays[0]
        total_time = 0
        diff = 1.5
        for _ in range(reps):
            total_time += time_function(
                lambda: tomopy_remove_outlier(data, FILTER_SIZE, diff)
            )
        operation_time = total_time / reps
        self.print_operation_times(
            total_time=total_time, operation_name="remove outlier", runs=reps
        )
        return operation_time
