
import numpy as np
import os

# FINN
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.visualization import showSrc, showInNetron
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.util.basic import make_build_dir
from finn.transformation.streamline import Streamline
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC, MoveScalarLinearPastInvariants
import finn.transformation.streamline.absorb as absorb
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.util.basic import pynq_part_map
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth

# QONNX
from qonnx.util.cleanup import cleanup
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.core.modelwrapper import ModelWrapper
import qonnx.core.onnx_exec as oxe
from qonnx.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
#from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.general import RemoveUnusedTensors
#from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.general import RemoveUnusedTensors
from qonnx.custom_op.registry import getCustomOp

# Variables
rel_path = "2025-03-21/16-47-24"
build_dir = os.environ["FINN_BUILD_DIR"]

# Cleanup
export_onnx_path = "tcn_v31.onnx"
export_onnx_path_cleaned = "qonnx_cleaned.onnx"
cleanup(export_onnx_path, out_file=export_onnx_path_cleaned)
#showInNetron(export_onnx_path)

# FINN format conversion
model = ModelWrapper(export_onnx_path_cleaned)
model = model.transform(ConvertQONNXtoFINN())
export_onnx_path_converted = "qonnx_cleaned_converted.onnx"
model.save(export_onnx_path_converted)

# Transforming onnx to be cleaner
export_onnx_path_converted_tidy = "qonnx_cleaned_converted_tidy.onnx"
model = ModelWrapper(export_onnx_path_converted)
model = model.transform(InferShapes())
model = model.transform(FoldConstants())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model = model.transform(InferDataTypes())
model = model.transform(RemoveStaticGraphInputs())
model = model.transform(Change3DTo4DTensors())
model = model.transform(MoveScalarLinearPastInvariants())
model = model.transform(Streamline())
model = model.transform(LowerConvsToMatMul())
#model = model.transform(MakeMaxPoolNHWC())
model = model.transform(MoveScalarLinearPastInvariants())
model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
#model = model.transform(ConvertBipolarMatMulToXnorPopcount())
model = model.transform(Streamline())
# absorb final add-mul nodes into TopK
#model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
model = model.transform(InferDataLayouts())
model = model.transform(RemoveUnusedTensors())
#model = model.transform(ConvertBipolarMatMulToXnorPopcount())
model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
# absorb final add-mul nodes into TopK
#model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
model = model.transform(RoundAndClipThresholds())
model = model.transform(MinimizeWeightBitWidth())
model = model.transform(MinimizeAccumulatorWidth())
# make sure the changed datatypes are propagated through the network
model = model.transform(InferDataTypes())
#model = model.transform(Streamline())
#model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
# bit of tidy-up
#model = model.transform(InferDataLayouts())
model = model.transform(RemoveUnusedTensors())
model.save(export_onnx_path_converted_tidy)

# Making final parts of model hw compatible
export_onnx_path_converted_tidy_to_hw = "qonnx_cleaned_converted_tidy_to_hw.onnx"
model = ModelWrapper(export_onnx_path_converted_tidy)
#model = model.transform(to_hw.InferBinaryMatrixVectorActivation())
# TopK to LabelSelect
#model = model.transform(to_hw.InferLabelSelectLayer())
model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
#model = model.transform(to_hw.InferVectorVectorActivation())
# input quantization (if any) to standalone thresholding

model = model.transform(to_hw.InferThresholdingLayer())
model = model.transform(to_hw.InferConvInpGen())


model.save(export_onnx_path_converted_tidy_to_hw)
#showInNetron(export_onnx_path_converted_tidy_to_hw)

# Separating top-level parent model from dataflow hw model 
export_onnx_path_converted_tidy_to_PARENT = "qonnx_cleaned_converted_tidy_to_hw_Parent.onnx"
model = ModelWrapper(export_onnx_path_converted_tidy_to_hw)
model = model.transform(absorb.AbsorbConsecutiveTransposes())
model = model.transform(RoundAndClipThresholds())
model = model.transform(MinimizeWeightBitWidth())
model = model.transform(MinimizeAccumulatorWidth())
model = model.transform(InferDataLayouts()) #EVEN
parent_model = model.transform(CreateDataflowPartition())
parent_model.save(export_onnx_path_converted_tidy_to_PARENT)
#showInNetron(export_onnx_path_converted_tidy_to_PARENT)

# Mapping to Pynq Z1
export_onnx_path_converted_tidy_to_hw_dataflow_hls = "qonnx_cleaned_converted_tidy_to_hw_dataflow_hls.onnx"
# node_types = {node.op_type for node in parent_model.graph.node}
# print("Available node types in ONNX model:", node_types)
sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
sdp_node = getCustomOp(sdp_node)
dataflow_model_filename = sdp_node.get_nodeattr("model")
model = ModelWrapper(dataflow_model_filename)
thresh_node = model.get_nodes_by_op_type("Thresholding")[0]
thresh_node_inst = getCustomOp(thresh_node)
thresh_node_inst.set_nodeattr("preferred_impl_style", "rtl")
pynq_board = "Pynq-Z1"
fpga_part = pynq_part_map[pynq_board]
target_clk_ns = 10
model = model.transform(SpecializeLayers(fpga_part))
model.save(export_onnx_path_converted_tidy_to_hw_dataflow_hls)

# Synthesising and creating driver for FPGA
model = ModelWrapper(export_onnx_path_converted_tidy_to_hw_dataflow_hls)
model = model.transform(ZynqBuild(platform = pynq_board, period_ns = target_clk_ns))
model = model.transform(MakePYNQDriver("zynq-iodma"))
export_onnx_path_converted_tidy_to_hw_dataflow_hls_POSTSYNTH = "/home/eveneiha/finn/workspace/finn/output_dir/qonnx_cleaned_converted_tidy_to_hw_dataflow_hls_POSTSYNTH.onnx"
model.save(export_onnx_path_converted_tidy_to_hw_dataflow_hls_POSTSYNTH)