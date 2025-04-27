
import os

# FINN
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.util.basic import pynq_part_map

from finn.transformation.streamline import Streamline
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
import finn.transformation.streamline.absorb as absorb


from qonnx.transformation.insert_topk import InsertTopK



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
from finn.transformation.qonnx.fold_quant_weights import FoldQuantWeights

from onnx import numpy_helper
# QONNX
from qonnx.util.cleanup import cleanup
from qonnx.custom_op.registry import getCustomOp
from qonnx.core.modelwrapper import ModelWrapper

from qonnx.transformation.general import *

from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants

from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC, MoveScalarLinearPastInvariants


from qonnx.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.general import RemoveUnusedTensors 
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.core.datatype import DataType





## --- CUSTOM STREAMLINGIN TEST --- 

from qonnx.transformation.base import Transformation
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.general import (
    ConvertDivToMul,
    ConvertSubToAdd,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.remove import RemoveIdentityOps

from finn.transformation.streamline.absorb import (
    Absorb1BitMulIntoConv,
    Absorb1BitMulIntoMatMul,
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
    FactorOutMulSignMagnitude,
)
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedAdd,
    CollapseRepeatedMul,
)
from finn.transformation.streamline.reorder import (
    MoveAddPastConv,
    MoveAddPastMul,
    MoveMulPastMaxPool,
    MoveScalarAddPastMatMul,
    MoveScalarLinearPastInvariants,
    MoveScalarMulPastConv,
    MoveScalarMulPastMatMul,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.streamline.sign_to_thres import ConvertSignToThres


import onnx
from onnx import numpy_helper

class StreamlineCustom(Transformation):
    """Custom streamline with debug prints after every step."""

    def apply(self, model):
        streamline_transformations = [
            ConvertSubToAdd(),
            ConvertDivToMul(),
            BatchNormToAffine(),
            ConvertSignToThres(),
            MoveMulPastMaxPool(),
            MoveScalarLinearPastInvariants(),
            AbsorbSignBiasIntoMultiThreshold(),
            MoveAddPastMul(),
            MoveScalarAddPastMatMul(),
            MoveAddPastConv(),
            MoveScalarMulPastMatMul(),
            MoveScalarMulPastConv(),
            MoveAddPastMul(),
            CollapseRepeatedAdd(),
            CollapseRepeatedMul(),
            MoveMulPastMaxPool(),
            AbsorbAddIntoMultiThreshold(),
            FactorOutMulSignMagnitude(),
            AbsorbMulIntoMultiThreshold(),
            Absorb1BitMulIntoMatMul(),
            Absorb1BitMulIntoConv(),
            RoundAndClipThresholds(),
        ]

        for trn in streamline_transformations:
            # print(f"\n🚀 Applying {trn.__class__.__name__}")
            model = model.transform(trn)

            # # After each transformation, print threshold stats
            # print(f"🔍 Threshold stats after {trn.__class__.__name__}:")
            # for init in model.graph.initializer:
            #     if "MultiThreshold" in init.name:
            #         th_arr = numpy_helper.to_array(init)
            #         print(f"- {init.name}: min={th_arr.min()}, max={th_arr.max()}, dtype={th_arr.dtype}")

            # Minor cleaning after each
            model = model.transform(RemoveIdentityOps())
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())
            model = model.transform(InferDataTypes())

        return (model, False)

## --- FINISHED CUSTOM STREAMLINING TEST ---























# --- Setup ---
pynq_board = "Pynq-Z1"
fpga_part = pynq_part_map[pynq_board]
target_clk_ns = 10

filename = './onnx/tcn_v41.onnx'
cleanup(filename, out_file=filename)




# --- Load and start ---
model = ModelWrapper(filename)
model.save("./onnx/step0a_modelwrapper.onnx")



# --- Graph cleanup ---
model = model.transform(ConvertQONNXtoFINN())
model.save("./onnx/step0_qonnx_to_finn.onnx")

model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model = model.transform(InferShapes())
model = model.transform(FoldConstants())




model = model.transform(RemoveStaticGraphInputs())
model = model.transform(GiveUniqueParameterTensors())
model = model.transform(SortGraph())

model = model.transform(ConvertSubToAdd())
model = model.transform(ConvertDivToMul())

model = model.transform(RemoveUnusedTensors())
model = model.transform(MovePadAttributeToTensor())
print("🔍 Thresholds after ConvertQONNXtoFINN:")
for init in model.graph.initializer:
    if "MultiThreshold" in init.name:
        th = model.get_initializer(init.name)
        print(f"- {init.name}: min={th.min()}, max={th.max()}, dtype={th.dtype}")
#model = model.transform(InsertTopK())
model.save("./onnx/step1_cleanup.onnx")



# --- Absorb moves ---
#model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
#model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
#model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
model.save("./onnx/step2_absorb.onnx")

# 🔥 Only now manually rescale thresholds!
print("🔍 Thresholds before streamline:")
for init in model.graph.initializer:
    if "MultiThreshold" in init.name:
        th = model.get_initializer(init.name)
        print(f"- {init.name}: min={th.min()}, max={th.max()}, dtype={th.dtype}")


model = model.transform(Streamline())

print("🧪 Thresholds after streamline:")
for init in model.graph.initializer:
    if "MultiThreshold" in init.name:
        th = model.get_initializer(init.name)
        print(f"- {init.name}: min={th.min()}, max={th.max()}")


model = model.transform(RoundAndClipThresholds())
model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
model = model.transform(RemoveUnusedTensors())
model = model.transform(InferDataLayouts())
model = model.transform(RemoveUnusedTensors())
model.save("./onnx/step2b_streamlined.onnx")



# --- Layout and MatMul lowering ---
model = model.transform(InferDataLayouts())
model = model.transform(RemoveIdentityOps())
model = model.transform(to_hw.InferGlobalAccPoolLayer())
model = model.transform(LowerConvsToMatMul())


model.save("./onnx/step3_lowered.onnx")

# --- Final preparations before HLS ---
model = model.transform(MinimizeWeightBitWidth())
model = model.transform(MinimizeAccumulatorWidth())
model = model.transform(InferDataTypes())
model = model.transform(RemoveUnusedTensors())
model = model.transform(RemoveCNVtoFCFlatten())
model.save("./onnx/step4_before_hw.onnx")


# --- Convert to HW layers ---
model = model.transform(to_hw.InferLookupLayer())
model = model.transform(to_hw.InferVectorVectorActivation())
model = model.transform(to_hw.InferThresholdingLayer())
model = model.transform(to_hw.InferConvInpGen())
model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
model = model.transform(to_hw.InferChannelwiseLinearLayer())
model = model.transform(RemoveUnusedTensors())
model.save("./onnx/step5_hw_layers.onnx")

model = model.transform(absorb.AbsorbConsecutiveTransposes())
model = model.transform(InferDataLayouts())
model = model.transform(RemoveUnusedTensors())
model.save("./onnx/step6b_transpose_cleaned.onnx")

# --- Prepare IP ---
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model = model.transform(PrepareIP(fpga_part, target_clk_ns))
model.save("./onnx/step6_prepared_ip.onnx")

# Absorb Mul into MultiThreshold if possible
#model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
model = model.transform(RemoveUnusedTensors())
model.save("./onnx/step7b_postmul_absorb.onnx")

# --- Partition into dataflow ---
model = model.transform(InferShapes())
model = model.transform(InferDataTypes())
model = model.transform(InferDataLayouts())
model = model.transform(RemoveUnusedTensors())
model.save("./onnx/step7_before_partition.onnx")


for idx, node in enumerate(model.graph.node):
    print(f"{idx}: {node.name} ({node.op_type}) - domain: {node.domain}")


parent_model = model.transform(CreateDataflowPartition())
parent_model.save("./onnx/step8_after_partition.onnx")

# --- Specialize and Insert FIFOs ---
sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
sdp_node = getCustomOp(sdp_node)
dataflow_model_filename = sdp_node.get_nodeattr("model")
model = ModelWrapper(dataflow_model_filename)

model = model.transform(SpecializeLayers(fpga_part))
model.save("./onnx/step9_specialized.onnx")


for node in model.graph.node:
    if node.op_type in ["StreamingFullyConnected_Batch", "StreamingFCLayer_Batch", "QuantizedMatrixVectorActivation"]:
        print(f"🔎 Node: {node.name}, op_type: {node.op_type}")
        weight_name = node.input[1]  # typically second input is the weight tensor
        weight_tensor = model.get_initializer(weight_name)

        print(f"  Shape: {weight_tensor.shape}")
        print(f"  Unique values: {np.unique(weight_tensor)}\n")

model = model.transform(InsertAndSetFIFODepths(fpgapart=fpga_part))
model.save("./onnx/step10_fifo_depths.onnx")

# --- Build bitfile and driver ---
model = model.transform(ZynqBuild(platform=pynq_board, period_ns=target_clk_ns))
model = model.transform(MakePYNQDriver("zynq-iodma"))

# --- Save final ---
final_export_path = "/home/eveneiha/finn/workspace/finn/output_dir/tcn_POSTSYNTH.onnx"
model.save(final_export_path)

print("\n🎯 FINN Transformations complete! Bitfile and driver ready for deployment.")
