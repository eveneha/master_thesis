import os
import sys 

# FINN
from finn.util.basic import pynq_part_map
import finn.transformation.streamline.absorb as absorb
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP

# QONNX
from qonnx.util.cleanup import cleanup
from qonnx.custom_op.registry import getCustomOp
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import *
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.infer_data_layouts import InferDataLayouts

#CUSTOM
from finn.slice_node.util import convert_node_io_to_nhwc, move_node_to_before
from slice_node.replace_slice_with_streaming_slice import ReplaceSliceWithStreamingSlice


import hashlib

# Add a depth counter
_transform_depth = 0

def hash_model(model):
    graph_str = str(model.graph)
    init_hashes = [
        hashlib.md5(model.get_initializer(init.name).tobytes()).hexdigest()
        for init in model.graph.initializer
    ]
    full_repr = graph_str + "".join(init_hashes)
    return hashlib.md5(full_repr.encode()).hexdigest()

# Save original method
ModelWrapper._orig_transform = ModelWrapper.transform

# Monkey-patch
def patched_transform(self, transformation, **kwargs):
    global _transform_depth
    _transform_depth += 1
    try:
        if _transform_depth == 1:  # Only track top-level transforms
            before = hash_model(self)
            result = self._orig_transform(transformation, **kwargs)
            after = hash_model(result)
            change = "✅ CHANGED" if before != after else "❌ NO CHANGE"
            print(f"[{change}] {transformation.__class__.__name__}")
            return result
        else:
            return self._orig_transform(transformation, **kwargs)
    finally:
        _transform_depth -= 1


ModelWrapper.transform = patched_transform

pynq_board = "Pynq-Z1"
fpga_part = pynq_part_map[pynq_board]
target_clk_ns = 10
filename= './onnx/tcn_v41_inf.onnx'

cleanup(filename, out_file=filename)
model = ModelWrapper(filename)
model.save('./onnx/00_after_cleanup.onnx')


model = model.transform(ConvertQONNXtoFINN())
model.save('./onnx/01_after_convertQONNXtoFINN.onnx')   
            
            
model = model.transform(GiveUniqueNodeNames())                                                                    
model = model.transform(GiveReadableTensorNames())  
model.save('./onnx/02tcn_before_qonnx_transforms.onnx')                                                                                                                        


model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
model = model.transform(Streamline()) # Apply a series of transformations to the model to make it more efficient.
model.save('./onnx/03_tcn_after_streamline.onnx')
model = model.transform(ReplaceSliceWithStreamingSlice()) ## <-- This is the custom op
model.save('./onnx/04_tcn_after_streaming_slice.onnx')
model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
model.save('./onnx/05_after_streaming_slice.onnx')

     
model = model.transform(InferDataLayouts())
model = model.transform(InferDataTypes())
model = model.transform(Streamline())
model.save('./onnx/06_tcn_after_qonnx_transforms.onnx')


model = model.transform(LowerConvsToMatMul())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model.save('./onnx/07_tcn_before_abs.onnx')


model = move_node_to_before(model, "Transpose_7", "MultiThreshold_4") ## <-- impossible to transpose into a streamingslice so move it after and do manually transpose on SS 
model = convert_node_io_to_nhwc(model, "StreamingSlice_0")
model.save('./onnx/8_tcn_after_transpose_rewire.onnx')


model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
model = model.transform(Streamline())
model = model.transform(InferDataLayouts())
model = model.transform(absorb.AbsorbConsecutiveTransposes())
model.save('./onnx/9_tcn_onnx_cleanup.onnx')


model = model.transform(to_hw.InferThresholdingLayer())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model.save('./onnx/10_tcn_beforeinferconvipgen.onnx')


model = model.transform(to_hw.InferConvInpGen())
model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
model = model.transform(MinimizeAccumulatorWidth())
model = model.transform(GiveUniqueNodeNames())                                                                            
model = model.transform(GiveReadableTensorNames())  
model.save('./onnx/11_tcn_reduce_acc_width.onnx')


model = model.transform(PrepareIP(fpga_part, 10))           
model.save('./onnx/13_tcn_before_part.onnx')

for idx, node in enumerate(model.graph.node):
    print(f"{idx}: {node.name} ({node.op_type}) - domain: {node.domain}")
    
parent_model = model.transform(CreateDataflowPartition())
parent_model.save('./onnx/14_tcn_after_part.onnx')


# # Mapping to Pynq Z1
sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
sdp_node = getCustomOp(sdp_node)
dataflow_model_filename = sdp_node.get_nodeattr("model")

# print(f"Path to dataflow model (inside partition): {dataflow_model_filename}")
model = ModelWrapper(dataflow_model_filename)

# --- Hardware Build ---
model.save('./onnx/15_tcn_before_specialize.onnx')
model = model.transform(SpecializeLayers(fpga_part))
model.save('./onnx/16_tcn_after_specialize.onnx')
model = model.transform(InsertAndSetFIFODepths(fpgapart=fpga_part))
model.save('./onnx/17_tcn_after_fifodepths.onnx')

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

print("\n" + "="*46)
print("           🎉 FINN BUILD COMPLETE 🎉")
print("="*46)
print("Your accelerator has been successfully built")
print("and is ready for deployment to your PYNQ board.")
print("="*46 + "\n")