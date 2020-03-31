import time

from imagingtester import ImagingTester, SIZES_SUBSET, DTYPE, N_RUNS, FILTER_SIZE
from cpu_imaging_filters import scipy_median_filter
from write_and_read_results import write_results_to_file, ARRAY_SIZES

LIB_NAME = "scipy"


def time_function(func):
    start = time.time()
    func()
    end = time.time()
    return end - start


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


# Create empty lists for storing results
add_arrays_results = []
background_correction_results = []
median_filter_results = []

for size in ARRAY_SIZES[:SIZES_SUBSET]:

    imaging_obj = CPUImplementation(size, DTYPE)
    median_filter_results.append(imaging_obj.timed_median_filter(N_RUNS))

write_results_to_file(
    [LIB_NAME, str(FILTER_SIZE)], "median filter", median_filter_results
)
