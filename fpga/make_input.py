import torch
import numpy as np
from pkgutil import get_data
import onnx.numpy_helper as nph
import matplotlib.pyplot as plt
from qonnx.core.modelwrapper import ModelWrapper

# Generate dummy input
dummy_input = torch.randn(766, 1, 1)
dummy_input = torch.tanh(dummy_input)  # Now all values are in (-1, 1)

# Initialize your model (make sure 'model' is defined correctly)
model = ModelWrapper("/home/eveneiha/finn/workspace/finn/output_dir/tcn_POSTSYNTH.onnx")

# Get the expected input tensor name and shape from the model
iname = model.graph.input[0].name
ishape = model.get_tensor_shape(iname)
print("Expected network input shape is " + str(ishape))

# # Quantize the input to 8-bit integer
scaled_input = dummy_input * 127.0
quantized_input = scaled_input.round().to(torch.int8)
np_input = quantized_input.cpu().numpy().reshape(ishape)


# Print the shape of the data
print("Data shape:", dummy_input.shape)

np_input = dummy_input.reshape(ishape)

# Rearrange the axes from (batch, channel, width, height) to (batch, width, channel, height)
# np_input = np.transpose(dummy_input.reshape(ishape), (0, 1, 2, 3))
print("Shape of created input is:", np_input.shape)  # This should print (1, 256, 1, 1)
np.save("/home/eveneiha/finn/workspace/fpga/input.npy", np_input)



