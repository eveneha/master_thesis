# file: replace_slice_test.py

import onnx
from qonnx.core.modelwrapper import ModelWrapper
from replace_slice_with_streaming_slice import ReplaceSliceWithStreamingSlice

# Step 1: Generate the test ONNX model

# Step 2: Load the generated model
model = ModelWrapper("test_slice.onnx")

# Step 3: Check original model
print("✅ Original graph nodes:")
for n in model.graph.node:
    print(f"  {n.name}: {n.op_type}")

# Step 4: Apply the transformation
model = model.transform(ReplaceSliceWithStreamingSlice())

# Step 5: Check modified model
print("\n✅ After ReplaceSliceWithStreamingSlice:")
for n in model.graph.node:
    print(f"  {n.name}: {n.op_type}")

# Step 6: Save the modified model
model.save("test_slice_replaced.onnx")
print("\n✅ Saved transformed model to test_slice_replaced.onnx")
