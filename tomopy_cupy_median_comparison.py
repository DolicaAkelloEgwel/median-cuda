import tomopy
import cupy as cp
import numpy as np
import time

from cupy_test_utils import create_padded_array, cupy_two_dim_median_filter
from write_and_read_results import write_results_to_file, ARRAY_SIZES
from imagingtester import DTYPE, SIZES_SUBSET, FILTER_SIZE, N_RUNS

N = 2
N_IMAGES = 5

cp.random.seed(19)
cp_data = cp.random.uniform(low=0, high=20, size=(N_IMAGES, N, N)).astype(DTYPE)
np_data = cp_data.get()
gpu_result = np.empty_like(np_data)

np_padded_data = create_padded_array(np_data, FILTER_SIZE // 2, "symmetric")
cp_padded_data = cp.array(np_padded_data)
diff = 0.5

cpu_result = tomopy.misc.corr.median_filter(np_data, FILTER_SIZE, axis=0)

for i in range(N_IMAGES):
    cp.cuda.runtime.deviceSynchronize()
    cupy_two_dim_median_filter(cp_data[i], cp_padded_data[i], FILTER_SIZE)
    cp.cuda.runtime.deviceSynchronize()
    gpu_result[i][:] = cp_data[i].get()

assert np.allclose(gpu_result, cpu_result)

tomopy_median_results = []

for size in ARRAY_SIZES[:SIZES_SUBSET]:

    arr = np.random.uniform(low=0, high=20, size=size).astype(DTYPE)
    start = time.time()
    tomopy.misc.corr.median_filter(arr, FILTER_SIZE, axis=0)
    end = time.time()
    tomopy_median_results.append(end - start)

write_results_to_file(
    ["tomopy", str(FILTER_SIZE)], "median filter", tomopy_median_results
)
