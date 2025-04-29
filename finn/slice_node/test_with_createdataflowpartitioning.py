from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from replace_slice_with_streaming_slice import ReplaceSliceWithStreamingSlice

# 1. Load your simple slice model
model = ModelWrapper("/home/eveneiha/finn/workspace/finn/onnx/tcn_beforePArt.onnx")

# 2. Replace Slice -> StreamingSlice
model = model.transform(ReplaceSliceWithStreamingSlice())
model.save("test_slice_streaming.onnx")
# 3. Apply CreateDataflowPartition (FINN will expect fpgadataflow nodes to work)
model = model.transform(CreateDataflowPartition())

# 4. Save result
model.save("test_slice_partitioned.onnx")

print("✅ Partitioned model saved as test_slice_partitioned.onnx")
