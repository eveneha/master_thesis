# test_codegen.py

import onnx
import onnx.helper as oh
from streaming_slice import StreamingSlice
from qonnx.core.modelwrapper import ModelWrapper
import numpy as np

# Define dummy ONNX node manually
input_name = "input0"
output_name = "output0"

slice_node = oh.make_node(
    "Slice",
    inputs=[input_name],
    outputs=[output_name],
    name="myslice"
)

# Create a dummy model wrapper with correct shape
model = ModelWrapper(onnx.helper.make_model(onnx.helper.make_graph(
    nodes=[],
    name="dummy_graph",
    inputs=[onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, [1, 1, 1000, 1])],
    outputs=[onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [1, 1, 665, 1])]
)))

# Fake StreamingSlice node
streaming_slice = StreamingSlice(slice_node)
streaming_slice.set_nodeattr("start_idx", 168)
streaming_slice.set_nodeattr("slice_length", 665)
streaming_slice.set_nodeattr("axis", 2)
streaming_slice.set_nodeattr("step", 1)

# Generate HLS project
streaming_slice.code_generation_ipgen(
    model=model,
    fpgapart="xc7z020clg400-1",
    clk=5.0
)

# After running this, check output:
print("✅ HLS project generated at:", streaming_slice.get_nodeattr("code_gen_dir_ipgen"))
