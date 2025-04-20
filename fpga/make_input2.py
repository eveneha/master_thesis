import torch 
import numpy as np

dummy_input = torch.randn(1, 766, 1, 1)
dummy_input = torch.tanh(dummy_input)
scaled_input = dummy_input * 127.0
quantized_input = scaled_input.round().to(torch.int8)
np_input = quantized_input.cpu().numpy().astype(np.int8)
np.save('/home/eveneiha/finn/workspace/fpga/input.npy', np_input)

