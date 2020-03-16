import tomopy
import cupy as cp
import numpy as np

from cupy_test_utils import create_padded_array, cupy_two_dim_remove_outliers
from imagingtester import DTYPE

size = 3
filter_height = size
filter_width = size
filter_size = (filter_height, filter_width)

N = 5

cp_data = cp.random.uniform(low=0, high=10, size=(N,N,N)).astype(DTYPE)
np_data = cp_data.get()
gpu_result = np.empty_like(np_data)

pad_height = filter_size[1] // 2
pad_width = filter_size[0] // 2
np_padded_data = create_padded_array(np_data, pad_width, pad_height, "reflect")
cp_padded_data = cp.array(np_padded_data)

diff = 1

cpu_result = tomopy.misc.corr.remove_outlier(np_data, diff, size)

for i in range(N):
    cupy_two_dim_remove_outliers(cp_data[i], cp_padded_data, diff, size, "light")
    gpu_result[i][:] = cp_data[i].get()

cp.cuda.runtime.deviceSynchronize()

print(cpu_result)
print(gpu_result)

print(np.isclose(gpu_result[0], cpu_result[0]))
assert np.allclose(gpu_result, cpu_result)