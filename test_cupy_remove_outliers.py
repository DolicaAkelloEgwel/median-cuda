import tomopy
import cupy as cp
import numpy as np

from cupy_test_utils import create_padded_array, cupy_two_dim_median_filter
from imagingtester import DTYPE

size = 3
N = 500

cp.random.seed(19)
cp_data = cp.random.uniform(low=0, high=20, size=(N, N, N)).astype(DTYPE)
np_data = cp_data.get()
gpu_result = np.empty_like(np_data)

np_padded_data = create_padded_array(np_data, size // 2, "symmetric")
cp_padded_data = cp.array(np_padded_data)
print("Padded array:")
print(cp_padded_data[0])
print(cp_padded_data.shape)

diff = 0.5

cpu_result = tomopy.misc.corr.median_filter(np_data, size)

for i in range(N):
    cp.cuda.runtime.deviceSynchronize()
    cupy_two_dim_median_filter(cp_data[i], cp_padded_data[i], size)
    cp.cuda.runtime.deviceSynchronize()
    gpu_result[i][:] = cp_data[i].get()

cp.cuda.runtime.deviceSynchronize()

print("Original data:")
print(np_data[0])
print("CPU:")
print(cpu_result[0])
print("GPU:")
print(gpu_result[0])

print("Is close: gpu result and input data")
print(np.isclose(gpu_result[0], np_data[0]))
print("Is close: cpu result and input data")
print(np.isclose(cpu_result[0], np_data[0]))

assert np.allclose(gpu_result, cpu_result)
