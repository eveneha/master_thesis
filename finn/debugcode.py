import numpy as np

x = np.load('finn_sample.npy').astype(np.int8)  # shape (1, 1000, 1, 1)

print('📦 Raw INT8 input values sent to FPGA:')
print(x.flatten()[:10])  # first 10 values for inspection