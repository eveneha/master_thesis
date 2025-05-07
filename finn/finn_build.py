
import os
import sys 
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

from slice_node.replace_slice_with_streaming_slice import ReplaceSliceWithStreamingSlice

from util import convert_node_io_to_nhwc, remove_node_and_rewire, update_node_attribute, move_node_to_before, set_transpose_output_shape, swap_streaming_slice_and_multithreshold

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


filename= './onnx/tcn_v41_inf.onnx'
# filename = "/home/eveneiha/finn/workspace/ml/model/pruned_model.onnx"

cleanup(filename, out_file=filename)

model = ModelWrapper(filename)
model.save('./onnx/00_after_cleanup.onnx')
model = model.transform(ConvertQONNXtoFINN())
model.save('./onnx/01_after_convertQONNXtoFINN.onnx')
#model.set_tensor_datatype("global_in", DataType["INT8"])

#model = model.transform(GiveUniqueNodeNames())                                                                    
#model = model.transform(GiveReadableTensorNames())  

# Transforming onnx to be cleaner
model = model.transform(InferShapes())                              
model = model.transform(FoldConstants())                             
model = model.transform(GiveUniqueNodeNames())                                                                    
model = model.transform(GiveReadableTensorNames())  
model.save('./onnx/02tcn_before_qonnx_transforms.onnx')                                                                                                                        
model = model.transform(RemoveStaticGraphInputs())                 
model = model.transform(GiveUniqueParameterTensors())                                                                    
model = model.transform(SortGraph())                                                                                                                                  
model = model.transform(ConvertSubToAdd())                 
model = model.transform(ConvertDivToMul())                        
model = model.transform(RemoveUnusedTensors())                                                                                 
model = model.transform(MovePadAttributeToTensor())  

model.save('./onnx/03_tcn_after_qonnx_transforms.onnx') # <--- FIRST VERSION THAT WORKS ON ONNXRUNTIME 
#model = model.transform(Streamline()) # Apply a series of transformations to the model to make it more efficient.




print("🧪 Thresholds Before streamline:")
for init in model.graph.initializer:
    if "MultiThreshold" in init.name:
        th = model.get_initializer(init.name)
        print(f"- {init.name}: min={th.min()}, max={th.max()}")

## must move final add above  MultiThreshold_10

model.save('./onnx/03_tcn_before_absorb.onnx')
model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
model.save('./onnx/_04_tcn_after_absorb.onnx')
model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
model.save('./onnx/04_where_did_bn_go.onnx')
#model = move_node_to_before(model, "Add_7","MultiThreshold_10")
model.save('./onnx/04_tcn_after_move_add.onnx')
#model = model.transform(absorb.AbsorbAddIntoMultiThreshold())



model = model.transform(Streamline()) # Apply a series of transformations to the model to make it more efficient.
model.save('./onnx/05_tcn_after_streamline.onnx')
model = model.transform(ReplaceSliceWithStreamingSlice()) ## <-- This is the custom op
# import sys 
model.save('./onnx/05_tcn_after_streaming_slice.onnx')
model = model.transform(absorb.AbsorbMulIntoMultiThreshold())

model.save('./onnx/06_after_streaming_slice.onnx')



print("🧪 Thresholds after streamline:")
for init in model.graph.initializer:
    if "MultiThreshold" in init.name:
        th = model.get_initializer(init.name)
        print(f"- {init.name}: min={th.min()}, max={th.max()}")



     
model.save('./onnx/07_tcn_before_globalavg.onnx')  
model = model.transform(InferDataLayouts())
model = model.transform(RemoveIdentityOps())
# model = model.transform(to_hw.InferGlobalAccPoolLayer())
model = model.transform(InferShapes())
model = model.transform(InferDataTypes())



model = model.transform(Streamline())
model.save('./onnx/08_tcn_after_qonnx_transforms.onnx')


## must transpose 



model = model.transform(LowerConvsToMatMul())

model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model.save('./onnx/09_tcn_before_abs.onnx')




# model = swap_streaming_slice_and_multithreshold(model , "StreamingSlice_0", "MultiThreshold_0")
# model.save('./onnx/09_inspect.onnx')
# model = set_transpose_output_shape(model, "Transpose_0", [1,1000,1,1])
# model = move_node_to_before(model, "Transpose_0", "MultiThreshold_0")
# model = convert_node_io_to_nhwc(model, "MultiThreshold_0")
# model = set_transpose_output_shape(model, "MultiThreshold_0", [1,1000,1,1])


model.save('./onnx/09_tcn_go.onnx')

#model = move_node_to_after(model, "StreamingSlice_0", "MultiThreshold_0")
# start_tensor = "Transpose_1_out0"
# end_tensor = "Transpose_3_out0"
# model = transpose_and_rewire_between_transposes(model, start_tensor, end_tensor)
model.save('./onnx/099_tnc.onnx')



#model = remove_node_and_rewire(model, "Transpose_1")

model = move_node_to_before(model, "Transpose_7", "MultiThreshold_5") ## <-- impossible to transpose into a streamingslice so move it after and do manually transpose on SS 
model = convert_node_io_to_nhwc(model, "StreamingSlice_0")


model.save('./onnx/10a_tcn_after_transpose_rewire.onnx')


#model = convert_node_io_to_nhwc(model, "StreamingSlice_1")

model.save('./onnx/10b_tcn_after_transpose_rewire.onnx')

model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
model = model.transform(Streamline())
model = model.transform(InferDataLayouts())
model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
model = model.transform(absorb.AbsorbConsecutiveTransposes())
model.save('./onnx/10__tcn_after_transpose_rewire.onnx')

#model = model.transform(absorb.AbsorbMulIntoMultiThreshold())


model.save('./onnx/11_tcn_after_abs.onnx')
model = model.transform(RoundAndClipThresholds())
model = model.transform(MinimizeWeightBitWidth())
model = model.transform(MinimizeAccumulatorWidth())
model = model.transform(InferDataTypes())
model = model.transform(RemoveUnusedTensors())
model = model.transform(RemoveCNVtoFCFlatten())


# Convertion to HLS 
model.save('./onnx/12_tcn_befor_hw.onnx')
model = model.transform(to_hw.InferLookupLayer())
model = model.transform(to_hw.InferVectorVectorActivation())
model = model.transform(to_hw.InferThresholdingLayer())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model = model.transform(absorb.AbsorbConsecutiveTransposes())

model.save('./onnx/13_tcn_after_hls.onnx')
# removing reduntand input parameters on streamingslice
model = model.transform(InferShapes())
model = model.transform(RemoveUnusedTensors())
model.save('./onnx/14_tcn_beforeinferconvipgen.onnx')
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
model.save('./onnx/18_tcn_before_prepareip.onnx')
model = model.transform(PrepareIP(fpga_part, 10))           
model.save('./onnx/18_afteR_prepare_IP.onnx')

model = model.transform(InferDataLayouts())
model = model.transform(RemoveUnusedTensors())
model.save('./onnx/19_tcn_beforePArt.onnx')


for idx, node in enumerate(model.graph.node):
    print(f"{idx}: {node.name} ({node.op_type}) - domain: {node.domain}")
    
parent_model = model.transform(CreateDataflowPartition())
parent_model.save('./onnx/20_tcn_after_oart.onnx')


# # Mapping to Pynq Z1
sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
sdp_node = getCustomOp(sdp_node)
dataflow_model_filename = sdp_node.get_nodeattr("model")

# print(f"Path to dataflow model (inside partition): {dataflow_model_filename}")
model = ModelWrapper(dataflow_model_filename)

# thresh_node = model.get_nodes_by_op_type("GlobalAccPool")[0]
# thresh_node_inst = getCustomOp(thresh_node)
# thresh_node_inst.set_nodeattr("preferred_impl_style", "rtl")




# --- Hardware Build ---

model.save('./onnx/21_tcn_before_specialize.onnx')

model = model.transform(SpecializeLayers(fpga_part))
model.save('./onnx/21_tcn_after_specialize.onnx')
#model = fix_streamingdwchapes(model)

# for node in model.graph.node:
#     if node.op_type == "StreamingSlice_hls":
#         inst = getCustomOp(node)
#         print("Node:", node.name)
#         print("ip_path:", inst.get_nodeattr("ip_path"))
#         print("impl_style:", inst.get_nodeattr("impl_style"))
#         print("backend:", inst.get_nodeattr("backend"))


model = model.transform(InsertAndSetFIFODepths(fpgapart=fpga_part))
model.save('./onnx/22_tcn_after_fifodepths.onnx')

# # Synthesising and creating driver for FPGA
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