# test_streaming_slice_execute_node.py

import numpy as np
from streaming_slice import StreamingSlice
from qonnx.core.modelwrapper import ModelWrapper
import onnx
import onnx.helper as oh

# Dummy ONNX model with one tensor
# We'll "fake" the inputs as needed
dummy_input_shape = [1, 1, 1000, 1]  # [N, C, T, 1]
input_name = "input0"
output_name = "output0"

# Create ONNX node (manually)
slice_node = oh.make_node(
    "Slice",
    inputs=[input_name],
    outputs=[output_name],
    name="myslice"
)

# Wrap into a fake StreamingSlice op
streaming_slice = StreamingSlice(slice_node)
streaming_slice.set_nodeattr("start_idx", 168)
streaming_slice.set_nodeattr("slice_length", 665)
streaming_slice.set_nodeattr("axis", 2)
streaming_slice.set_nodeattr("step", 1)

# Dummy context (like during FINN execution)
dummy_input = np.arange(1000).reshape(1, 1, 1000, 1).astype(np.float32)

context = {input_name: dummy_input}

# Execute
streaming_slice.execute_node(context, None)

# Get output
result = context[output_name]

print("Input shape:", dummy_input.shape)
print("Output shape:", result.shape)
print("First 10 output values:", result.flatten()[:10])
print("Last 10 output values:", result.flatten()[-10:])
