from cupy_test_utils import CupyImplementation, LIB_NAME
from imagingtester import SIZES_SUBSET, DTYPE, N_RUNS, FILTER_SIZE
from write_and_read_results import ARRAY_SIZES, write_results_to_file

# Create empty lists for storing results
median_filter_results = []

for size in ARRAY_SIZES[:SIZES_SUBSET]:

    imaging_obj = CupyImplementation(size, DTYPE)

    avg_mf = imaging_obj.timed_async_median_filter(N_RUNS, FILTER_SIZE)

    if avg_mf > 0:
        median_filter_results.append(avg_mf)

    write_results_to_file([LIB_NAME, "async"], "median filter", median_filter_results)
