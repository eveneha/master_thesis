import numpy as np
import torch 
np_input = np.load("input.npy")
print("Min value:", np_input.min())
print("Max value:", np_input.max())


dummy_input = torch.randn(256, 1, 1)
print("Min value:", dummy_input)