
import os

# FINN
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.util.basic import pynq_part_map

from finn.transformation.streamline import Streamline
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
import finn.transformation.streamline.absorb as absorb

from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from qonnx.transformation.remove import RemoveIdentityOps
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP


# QONNX
from qonnx.util.cleanup import cleanup
from qonnx.custom_op.registry import getCustomOp
from qonnx.core.modelwrapper import ModelWrapper

from qonnx.transformation.general import *

from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants

from qonnx.transformation.infer_datatypes import InferDataTypes


from qonnx.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.general import RemoveUnusedTensors 
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.core.datatype import DataType

pynq_board = "Pynq-Z1"
fpga_part = pynq_part_map[pynq_board]
target_clk_ns = 10


filename= './onnx/tcn_v41.onnx'
# filename = "/home/eveneiha/finn/workspace/ml/model/pruned_model.onnx"

cleanup(filename, out_file=filename)

model = ModelWrapper(filename)
model = model.transform(ConvertQONNXtoFINN())
model.set_tensor_datatype("global_in", DataType["INT8"])

# Transforming onnx to be cleaner
model = model.transform(InferShapes())                              # Ensure every tensor in the model has a specified shape (ValueInfo).

model = model.transform(FoldConstants())                            # Replace the output of a node with const-only inputs with a precomputed
                                                                    #   result. Skip any op types given in exclude_op_types.
model = model.transform(GiveUniqueNodeNames())                      # Give unique names to each node in the graph using enumeration, starting
                                                                    #   with given prefix (if specified in the constructor).
model = model.transform(GiveReadableTensorNames())                  # Give more human-readable names to all internal tensors. You should
                                                                    #   apply GiveUniqueNodeNames prior to this transform to avoid empty node names,
                                                                    #   as the readable names are based on the node names.
model = model.transform(RemoveStaticGraphInputs())                  # Remove any top-level graph inputs that have initializers.
model = model.transform(GiveUniqueParameterTensors())               # Make every parameter tensor unique. The aim is to avoid affecting
                                                                    #   other nodes apart from the one the system is currently operating on.
model = model.transform(SortGraph())                                # Returns the model with its node list sorted topologically.
                                                                    #   Any ONNX graph to be executed must have a topologically sorted node list,
                                                                    #   as dictated by the ONNX standard.
model = model.transform(ConvertSubToAdd())                          # Convert subtract-a-constant nodes to add-a-constant nodes.  
model = model.transform(ConvertDivToMul())                          # Convert divide by constant nodes to multiply by constant nodes.

                 
#model = model.transform(Change3DTo4DTensors())

# def count_nodes_and_initializers(finn_model):
#     g = finn_model.graph
#     return len(g.node), len(g.initializer)

# model_before = model
# node_count_before, init_count_before = count_nodes_and_initializers(model_before)

# model_after = model_before.transform(RemoveUnusedTensors())
# node_count_after, init_count_after = count_nodes_and_initializers(model_after)

# print(f"Nodes: {node_count_before} --> {node_count_after}")
# print(f"Initializers: {init_count_before} --> {init_count_after}")


model = model.transform(RemoveUnusedTensors())                      # Remove any unused tensors in the graph by removing any initializers,

                                                                    #  ValueInfo and tensor annotations associated with it. Unused tensors do not
                                                                    #  appear as any input/output for any graph nodes.
model = model.transform(MovePadAttributeToTensor())                 # Move padding info from attribute into input tensor for Pad nodes.  
model = model.transform(Streamline())                               # Apply a series of transformations to the model to make it more efficient.
model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
model.save('./onnx/tcn_before_globalavg.onnx')  
model = model.transform(InferDataLayouts())
model = model.transform(RemoveIdentityOps())
model = model.transform(to_hw.InferGlobalAccPoolLayer())
model = model.transform(InferShapes())
model = model.transform(InferDataTypes())


model.save('./onnx/tcn_after_qonnx_transforms.onnx')


model = model.transform(Streamline())
model = model.transform(LowerConvsToMatMul())


model.save('./onnx/tcn_before_abs.onnx')


model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
model = model.transform(Streamline())
model = model.transform(InferDataLayouts())
model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())

model = model.transform(absorb.AbsorbConsecutiveTransposes())

model.save('./onnx/tcn_after_abs.onnx')
model = model.transform(RoundAndClipThresholds())
model = model.transform(MinimizeWeightBitWidth())
model = model.transform(MinimizeAccumulatorWidth())
model = model.transform(InferDataTypes())

model = model.transform(RemoveUnusedTensors())


# Making final parts of model hw compatible
model = model.transform(RemoveCNVtoFCFlatten())


# Convertion to HLS 
model.save('./onnx/tcn_befor_hw.onnx')

for node in model.graph.node:
    if node.name == "Gather_0":
        print(f"🔎 Node: {node.name} - op_type: {node.op_type}", flush=True)

model = model.transform(to_hw.InferLookupLayer())


model = model.transform(to_hw.InferVectorVectorActivation())
model = model.transform(to_hw.InferThresholdingLayer())
model = model.transform(to_hw.InferConvInpGen())
model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())

model = model.transform(RoundAndClipThresholds())
model = model.transform(MinimizeWeightBitWidth())
model = model.transform(MinimizeAccumulatorWidth())
model = model.transform(InferDataLayouts()) #EVEN
model = model.transform(absorb.AbsorbConsecutiveTransposes())

model = model.transform(to_hw.InferChannelwiseLinearLayer())
# model = model.transform(InsertFIFO(create_shallow_fifos=True))

model = model.transform(RemoveUnusedTensors())


print(model.get_tensor_layout("MultiThreshold_11_out0"))
print(model.get_tensor_layout("GlobalAccPool_0_out0"))
model = model.transform(InferShapes())
model = model.transform(InferDataTypes())

# BEFORE partitioning explicitly:
model = model.transform(InferShapes())
model = model.transform(InferDataTypes())
model = model.transform(InferDataLayouts())
model = model.transform(InferShapes())
# Explicit FIFO insertion to break cyclic dependency:

model = model.transform(RemoveUnusedTensors())

    
# Finally
model = model.transform(GiveUniqueNodeNames())                      # Give unique names to each node in the graph using enumeration, starting
                                                                    #   with given prefix (if specified in the constructor).
model = model.transform(GiveReadableTensorNames())  

model = model.transform(PrepareIP('Pynq-Z1', 10))             # Important



# BEFORE partitioning explicitly:
for idx, node in enumerate(model.graph.node):
    print(f"{idx}: {node.name}")
for node in model.graph.node[33:76]:
    print(f"{node.name}: inputs = {node.input}, outputs = {node.output}")

model = model.transform(InferDataLayouts())
# Explicit FIFO insertion to break cyclic dependency:

model = model.transform(RemoveUnusedTensors())

inputs_to_keep = [i for i in model.graph.input if i.name != "Thresholding_0_param0"]
del model.graph.input[:]
model.graph.input.extend(inputs_to_keep)
print("✅ Removed Thresholding_0_param0 from model.graph.input (if it existed)")

# This will tell us if it's missing as initializer
print("Thresholding_0_param0 in initializers:", "Thresholding_0_param0" in [init.name for init in model.graph.initializer])
for i in model.graph.input:
    if i.name == "Thresholding_0_param0":
        print("FOUND IN graph.input!")
# This should return None
print("Producer of Thresholding_0_param0:", model.find_producer("Thresholding_0_param0"))
print("Is in initializers:", "Thresholding_0_param0" in [x.name for x in model.graph.input])
for node in model.graph.node:
    if node.op_type == "StreamingFIFO" and "Thresholding_0_param0" in node.input:
        print(f"BAD FIFO node: {node.name} is using Thresholding_0_param0 as input!")

print("=== Operator domains ===")
for node in model.graph.node:
    print(f"{node.name}: domain = {node.domain}")



model.save('./onnx/tcn_beforePArt.onnx')


for idx, node in enumerate(model.graph.node):
    print(f"{idx}: {node.name} ({node.op_type}) - domain: {node.domain}")
    
for node in model.graph.node:
    print(f"Node {node.name} inputs: {node.input}, outputs: {node.output}")



parent_model = model.transform(CreateDataflowPartition())
parent_model.save('./onnx/tcn_after_oart.onnx')


# # Mapping to Pynq Z1
sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
sdp_node = getCustomOp(sdp_node)
dataflow_model_filename = sdp_node.get_nodeattr("model")
model = ModelWrapper(dataflow_model_filename)

# thresh_node = model.get_nodes_by_op_type("GlobalAccPool")[0]
# thresh_node_inst = getCustomOp(thresh_node)
# thresh_node_inst.set_nodeattr("preferred_impl_style", "rtl")




# --- Hardware Build ---

model = model.transform(SpecializeLayers(fpga_part))
model.save('./onnx/tcn_after_specialize.onnx')


model = model.transform(InsertAndSetFIFODepths(fpgapart=fpga_part))
model.save('./onnx/tcn_after_fifodepths.onnx')

# # Synthesising and creating driver for FPGA
model = ModelWrapper('./onnx/tcn_after_fifodepths.onnx')
model = model.transform(ZynqBuild(platform = pynq_board, period_ns = target_clk_ns))
model = model.transform(MakePYNQDriver("zynq-iodma"))
export_onnx_path_converted_tidy_to_hw_dataflow_hls_POSTSYNTH = "/home/eveneiha/finn/workspace/finn/output_dir/tcn_POSTSYNTH.onnx"
model.save(export_onnx_path_converted_tidy_to_hw_dataflow_hls_POSTSYNTH)


# --- Deployment ---
model = ModelWrapper("/home/eveneiha/finn/workspace/finn/output_dir/tcn_POSTSYNTH.onnx")
sdp_node_middle = getCustomOp(model.graph.node[1])
postsynth_layers = sdp_node_middle.get_nodeattr("model")

model = ModelWrapper(postsynth_layers)
model.model.metadata_props


# Get the directory path from the model metadata
directory = model.get_metadata_prop("/home/eveneiha/finn/finn_host_build_dir/vivado_pynq_proj")

files = os.listdir(directory)
print(files)

print("\n" + "="*40)
print("        🎉 FINN BUILD COMPLETE 🎉")
print("="*40)
print("Your accelerator has been successfully built")
print("and is ready for deployment to your PYNQ board.")
print("="*40 + "\n")