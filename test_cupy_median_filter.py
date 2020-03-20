import cupy as cp
import numpy as np
from cupy_test_utils import (
    CupyImplementation,
    LIB_NAME,
    free_memory_pool,
    REFLECT_MODE,
    cupy_three_dim_median_filter,
)
from imagingtester import (
    USE_CUPY_NONPINNED_MEMORY,
    SIZES_SUBSET,
    DTYPE,
    N_RUNS,
    FILTER_SIZE,
)
from cpu_imaging_filters import scipy_median_filter
from write_and_read_results import ARRAY_SIZES, write_results_to_file

if USE_CUPY_NONPINNED_MEMORY:
    pinned_memory_mode = [True, False]
else:
    pinned_memory_mode = [True]

# These need to be odd values and need to be equal
filter_size = 3

test_size = 50

cp_data = cp.random.uniform(
    low=0, high=10, size=(test_size, test_size, test_size)
).astype(DTYPE)
np_data = cp_data.get()

# Create a padded array in the GPU
pad_size = filter_size // 2
padded_data = cp.pad(
    cp_data,
    pad_width=((0, 0), (pad_size, pad_size), (pad_size, pad_size)),
    mode=REFLECT_MODE,
)

# Run the median filter on the GPU
cupy_three_dim_median_filter(
    data=cp_data, padded_data=padded_data, filter_size=filter_size
)
# Run the scipy median filter
scipy_median_filter(np_data, size=filter_size)
# Check that the results match
cp_result = cp_data.get()
assert np.allclose(np_data, cp_result)

del cp_data
del padded_data
free_memory_pool()

for use_pinned_memory in pinned_memory_mode:

    # Create empty lists for storing results
    median_filter_results = []

    if use_pinned_memory:
        memory_string = "with pinned memory"
    else:
        memory_string = "without pinned memory"

    for size in ARRAY_SIZES[:SIZES_SUBSET]:

        imaging_obj = CupyImplementation(size, DTYPE, use_pinned_memory)

        avg_mf = imaging_obj.timed_three_dim_median_filter(N_RUNS, FILTER_SIZE)

        if avg_mf > 0:
            median_filter_results.append(avg_mf)

        write_results_to_file(
            [LIB_NAME, memory_string], "median filter", median_filter_results
        )
