# file: make_test_slice_model.py

import onnx
import numpy as np
import onnx.helper as oh
import onnx.numpy_helper as np_helper

# Create input and output tensors
input_tensor = oh.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1, 1, 100, 1])
output_tensor = oh.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1, 1, 20, 1])

# Create Slice attributes manually
starts = oh.make_tensor(name="starts", data_type=onnx.TensorProto.INT64, dims=[1], vals=[10])
ends = oh.make_tensor(name="ends", data_type=onnx.TensorProto.INT64, dims=[1], vals=[30])
axes = oh.make_tensor(name="axes", data_type=onnx.TensorProto.INT64, dims=[1], vals=[2])
steps = oh.make_tensor(name="steps", data_type=onnx.TensorProto.INT64, dims=[1], vals=[1])

# Initializers
initializers = [starts, ends, axes, steps]

# Create Slice node
slice_node = oh.make_node(
    'Slice',
    inputs=['input', 'starts', 'ends', 'axes', 'steps'],
    outputs=['output'],
    name='my_slice_node'
)

# Assemble the graph
graph = oh.make_graph(
    nodes=[slice_node],
    name='SliceGraph',
    inputs=[input_tensor],
    outputs=[output_tensor],
    initializer=initializers
)

# Create the model
model = oh.make_model(graph, producer_name='slice-test')

# Save model
onnx.save(model, 'test_slice.onnx')

print("✅ Test ONNX model with Slice node saved as test_slice.onnx")
