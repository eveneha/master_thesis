
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
#model.set_tensor_datatype("global_in", DataType["INT8"])

# Transforming onnx to be cleaner
model = model.transform(InferShapes())                              
model = model.transform(FoldConstants())                             
model = model.transform(GiveUniqueNodeNames())                                                                    
model = model.transform(GiveReadableTensorNames())                                                                                                                          
model = model.transform(RemoveStaticGraphInputs())                 
model = model.transform(GiveUniqueParameterTensors())                                                                    
model = model.transform(SortGraph())                                                                                                                                  
model = model.transform(ConvertSubToAdd())                 
model = model.transform(ConvertDivToMul())                        
model = model.transform(RemoveUnusedTensors())                                                                                 
model = model.transform(MovePadAttributeToTensor())  


print("🧪 Thresholds Before streamline:")
for init in model.graph.initializer:
    if "MultiThreshold" in init.name:
        th = model.get_initializer(init.name)
        print(f"- {init.name}: min={th.min()}, max={th.max()}")

model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
model = model.transform(Streamline()) # Apply a series of transformations to the model to make it more efficient.

print("🧪 Thresholds after streamline:")
for init in model.graph.initializer:
    if "MultiThreshold" in init.name:
        th = model.get_initializer(init.name)
        print(f"- {init.name}: min={th.min()}, max={th.max()}")
        
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
model = model.transform(RemoveUnusedTensors())
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
model = model.transform(GiveUniqueNodeNames())                     
                                                                   
model = model.transform(GiveReadableTensorNames())  

model = model.transform(PrepareIP(fpga_part, 10))           
model.save('./onnx/afteR_prepare_IP.onnx')


model = model.transform(InferDataLayouts())
model = model.transform(RemoveUnusedTensors())
model.save('./onnx/tcn_beforePArt.onnx')


for idx, node in enumerate(model.graph.node):
    print(f"{idx}: {node.name} ({node.op_type}) - domain: {node.domain}")
    
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