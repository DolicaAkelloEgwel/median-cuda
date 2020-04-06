import tomopy
import cupy as cp
import numpy as np

from cupy_test_utils import (
    CupyImplementation,
    create_padded_array,
    cupy_two_dim_remove_outliers,
)
from imagingtester import DTYPE, SIZES_SUBSET, FILTER_SIZE, N_RUNS
from write_and_read_results import write_results_to_file, ARRAY_SIZES

size = 3
IMAGES = 3
N = 200
LIB_NAME = "cupy"

cp.random.seed(19)
cp_data = cp.random.uniform(low=0, high=20, size=(IMAGES, N, N)).astype(DTYPE)
np_data = cp_data.get()
gpu_result = np.empty_like(np_data)

np_padded_data = create_padded_array(np_data, size // 2, "symmetric")
cp_padded_data = cp.array(np_padded_data)
diff = 0.5

cpu_result = tomopy.misc.corr.remove_outlier(np_data, diff, size)

for i in range(IMAGES):
    cp.cuda.runtime.deviceSynchronize()
    cupy_two_dim_remove_outliers(cp_data[i], cp_padded_data[i], diff, size, "light")
    cp.cuda.runtime.deviceSynchronize()
    gpu_result[i][:] = cp_data[i].get()

cp.cuda.runtime.deviceSynchronize()

assert np.allclose(gpu_result, cpu_result)

remove_outlier_results = []

for size in ARRAY_SIZES[:SIZES_SUBSET]:

    imaging_obj = CupyImplementation(size, DTYPE)

    avg_ro = imaging_obj.timed_async_remove_outlier(N_RUNS, FILTER_SIZE)

    if avg_ro > 0:
        remove_outlier_results.append(avg_ro)

    write_results_to_file(
        [LIB_NAME, "async", str(FILTER_SIZE)], "remove outlier", remove_outlier_results
    )
