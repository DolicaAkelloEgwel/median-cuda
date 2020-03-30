from cupy_test_utils import CupyImplementation, LIB_NAME, REFLECT_MODE, cupy_two_dim_median_filter, free_memory_pool
import cupy as cp
import numpy as np
import scipy.ndimage as scipy_ndimage
from imagingtester import SIZES_SUBSET, DTYPE, N_RUNS, FILTER_SIZE
from write_and_read_results import ARRAY_SIZES, write_results_to_file
from cpu_imaging_filters import scipy_median_filter

# Create empty lists for storing results
median_filter_results = []

# These need to be odd values and need to be equal
filter_size = 3
test_size = 555


cp_data = cp.random.uniform(
    low=0, high=10, size=(test_size, test_size)
).astype(DTYPE)
np_data = cp_data.get()

# Create a padded array in the GPU
pad_size = filter_size // 2
padded_data = cp.pad(
    cp_data,
    pad_width=((pad_size, pad_size), (pad_size, pad_size)),
    mode=REFLECT_MODE,
)

# Run the median filter on the GPU
cupy_two_dim_median_filter(
    data=cp_data, padded_data=padded_data, filter_size=filter_size
)
print("Original data:")
print(np_data)
# Run the scipy median filter
cpu_result = scipy_ndimage.median_filter(np_data, size=(filter_size,filter_size), mode="mirror")
# Check that the results match
cp_result = cp_data.get()
print("GPU result:")
print(cp_data)
print("CPU result:")
print(cpu_result)
assert np.allclose(np_data, cp_result)

del cp_data
del padded_data
free_memory_pool()

for size in ARRAY_SIZES[:SIZES_SUBSET]:

    imaging_obj = CupyImplementation(size, DTYPE)

    avg_mf = imaging_obj.timed_async_median_filter(N_RUNS, FILTER_SIZE)

    if avg_mf > 0:
        median_filter_results.append(avg_mf)

    write_results_to_file([LIB_NAME, "async", str(FILTER_SIZE)], "median filter", median_filter_results)
