from qonnx.core.modelwrapper import ModelWrapper

model = ModelWrapper("/home/eveneiha/finn/workspace/finn/output_dir/tcn_POSTSYNTH.onnx")

from qonnx.custom_op.registry import getCustomOp

for node in model.graph.node:
    if node.op_type == "StreamingFIFO":
        print(f"{node.name}: depth = {getCustomOp(node).get_nodeattr('depth')}")
