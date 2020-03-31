import time

from imagingtester import SIZES_SUBSET, DTYPE, N_RUNS, FILTER_SIZE
from cpu_imaging_filters import CPUImplementation
from write_and_read_results import write_results_to_file, ARRAY_SIZES

LIB_NAME = "tomopy"

remove_outlier_results = []

for size in ARRAY_SIZES[:SIZES_SUBSET]:

    imaging_obj = CPUImplementation(size, DTYPE)
    remove_outlier_results.append(imaging_obj.timed_remove_outlier(N_RUNS))

write_results_to_file(
    [LIB_NAME, str(FILTER_SIZE)], "remove outlier", remove_outlier_results
)
