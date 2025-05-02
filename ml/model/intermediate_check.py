
# --- Imports ---
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
#import scipy.signal
#import pywt

#from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder
#from imblearn.over_sampling import SMOTE

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
from torchinfo import summary
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat, Int8WeightPerChannelFloat, Uint8ActPerTensorFloat


import torch
import torch.nn as nn
import numpy as np
import os
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.core.onnx_exec import execute_onnx
from torch.utils.data import Dataset, DataLoader # Assuming needed if test_dataloader isn't pre-loaded
from onnx import shape_inference
# --- Brevitas Imports (Keep only if needed for model definition) ---
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat, Int8WeightPerChannelFloat # Use Uint8WeightPerChannelFloat if needed
from brevitas.quant import Uint8ActPerTensorFloat, Int8ActPerTensorFloat
from onnx import shape_inference
from qonnx.core.modelwrapper import ModelWrapper
import logging
logging.getLogger("qonnx.core.datatype").setLevel(logging.WARNING)

# ==============================================================================
# --- Configuration - CHOOSE MODEL AND PATHS HERE ---
# ==============================================================================

# Options: "original" or "inference"
MODEL_CHOICE = "inference"

# Path to the ONNX model file you want to simulate
# This should correspond to the processed version of the chosen PyTorch model
# (e.g., the ONNX just before partitioning for the chosen architecture)
# **MAKE SURE THIS ONNX FILE HAS INTERMEDIATE TENSORS/NODES CORRESPONDING TO THE PYTORCH LAYERS YOU WANT TO COMPARE**
ONNX_MODEL_PATH = "/home/eveneiha/finn/workspace/finn/onnx/000_after_cleanup.onnx" # <--- PATH TO ONNX TO TEST

# Path to the PyTorch checkpoint (.pth file) containing the weights
# Make sure these weights correspond to the chosen MODEL_CHOICE architecture
PYTORCH_WEIGHTS_PATH = "/home/eveneiha/finn/workspace/ml/model/tcn_model_v41_avg_pool.pth" # <--- PATH TO WEIGHTS

# Path to the input data (.pt file)
INPUT_DATA_PATH = "/home/eveneiha/finn/workspace/ml/data/preprocessed/test.pt"

NUM_SAMPLES_TO_TEST = 1 # Set this back to 20 or more if you want to test multiple samples
NUM_OUTPUT_CLASSES = 5

# Slicing parameters (only used if MODEL_CHOICE is "inference" for ONNX input)
# The PyTorch model should handle slicing internally if needed.
INPUT_SLICE_START = 168
INPUT_SLICE_END = 833 # This is exclusive, so it slices up to index 832

# Tolerances for comparison (used for final output and intermediate outputs)
ATOL = 1e-5
RTOL = 1e-4

# ==============================================================================
# --- PyTorch Model Definitions (Keep YOUR definitions here) ---
# ==============================================================================
# (Keeping your original model definitions as they were, ensuring they match the checkpoint)

class TemporalBlock2d(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, stride, dropout=0.2, slice_name= None):
        super(TemporalBlock2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride

        self.conv1 = qnn.QuantConv2d(
            n_inputs, n_outputs,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(0, 0),  # no padding – only compute valid outputs
            dilation=(dilation, 1),
            weight_quant=Int8WeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat,
            weight_bit_width=8,
            act_bit_width=8,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = qnn.QuantConv2d(
            n_outputs, n_outputs,
            kernel_size=(kernel_size, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(dilation, 1),
            weight_quant=Int8WeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat,
            weight_bit_width=8,
            act_bit_width=8,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        self.relu = qnn.QuantReLU(return_quant_tensor=False, act_quant=Uint8ActPerTensorFloat, output_quant=Int8ActPerTensorFloat)



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x


class TCN2d(nn.Module):
    def __init__(self, custom_blocks: list, num_outputs: int):
        super(TCN2d, self).__init__()
        self.temporal_blocks = nn.ModuleList(custom_blocks)
        last_out_channels = custom_blocks[-1].conv2.out_channels

        # We also need a 1x1 conv to get to num_outputs
        self.fc = qnn.QuantConv2d(
            in_channels=last_out_channels,
            out_channels=num_outputs,
            kernel_size=(1, 1),
            weight_quant=Int8WeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat,
            weight_bit_width=8,
            act_bit_width=8,
            bias=False
        )

    def forward(self, x): # Input is float
        for block in self.temporal_blocks:
            x = block(x)
        # Based on your original code snippet's fc usage, it seems like x
        # is potentially already reduced spatially or you just take a slice.
        # For THIS script, we'll assume the original model passes the potentially
        # large spatial output of the last block to FC, and the FC reduces spatial.
        # If your *actual* original model forward pass was different, adjust here.
        # The version below matches the inference model's approach more closely
        # by selecting a specific spatial index 84:85 before the FC.
        # If your *actual* original model forward pass was different, update this.
        x = x[:, :, 84:85, :] # Assuming this slice is also part of the original forward for v41/avg_pool
        x = self.fc(x)
        x = x.value if hasattr(x, 'value') else x # Get value from QuantTensor
        x_reshaped = x.reshape(x.size(0), -1)
        return x_reshaped


# --- Redefinition of TCN to TCN2d_inf for inference ---

class SliceSelectorStep(nn.Module):
    def __init__(self, step = 1):
        super().__init__()
        self.step = step

    def forward(self, x):
        # Based on TemporalBlock2d_inf __init__ calls in the script,
        # only block2_inf has this with step=4.
        # It seems to apply a slice [:65:4] to the temporal dimension.
        return x[:, :, :65:self.step, :]


class TemporalBlock2d_inf(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dropout=0.2, use_stride = False, last_block = False):
        super(TemporalBlock2d_inf, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.last_block = last_block # This determines if the slice_selector is added

        self.conv1 = qnn.QuantConv2d(
            n_inputs, n_outputs,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1), # Stride applied here based on input `stride`
            padding=(0, 0),
            weight_quant=Int8WeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat,
            weight_bit_width=8,
            act_bit_width=8,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)

        # Stride for conv2 seems determined by `use_stride` flag, not input `stride`
        conv2_stride = stride if use_stride else 1 # This makes block1 conv2 stride 2, others 1. Matches my interpretation above.

        self.conv2 = qnn.QuantConv2d(
            n_outputs, n_outputs,
            kernel_size=(kernel_size, 1),
            stride=(conv2_stride, 1), # Stride applied here based on `use_stride`
            padding=(0, 0),
            weight_quant=Int8WeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat,
            weight_bit_width=8,
            act_bit_width=8,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(n_outputs)
        self.relu = qnn.QuantReLU(return_quant_tensor=False, act_quant=Uint8ActPerTensorFloat, output_quant=Int8ActPerTensorFloat)
        self.dropout2 = nn.Dropout(dropout)

        # Slice selector is ONLY added to the block if last_block is True
        # And its step is hardcoded to 4 here, and it applies slice [:65:4]
        self.slice_selector_step = SliceSelectorStep(4) if last_block else None


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        # Order from your code: conv2 -> (slice if exists) -> bn2 -> relu -> dropout2
        x = self.slice_selector_step(x) if self.slice_selector_step is not None else x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x


class TCN2d_inf(nn.Module):
    def __init__(self, custom_blocks: list, num_outputs: int):
        super(TCN2d_inf, self).__init__()
        self.temporal_blocks = nn.ModuleList(custom_blocks)
        last_out_channels = custom_blocks[-1].conv2.out_channels

        self.fc = qnn.QuantConv2d(
            in_channels=last_out_channels,
            out_channels=num_outputs,
            kernel_size=(1, 1),
            weight_quant=Int8WeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat,
            weight_bit_width=8,
            act_bit_width=8,
            bias=False
        )

        # Input quant layer - keeping commented as in original
        # self.inp_quant = qnn.QuantIdentity(
        #     bit_width=8,
        #     act_quant=Int8ActPerTensorFloat,
        #     return_quant_tensor=True
        # )

        # self.out_quant = qnn.QuantIdentity(
        #     bit_width=8,
        #     act_quant=Int8ActPerTensorFloat,
        #     return_quant_tensor=True
        # )


    def forward(self, x):
        # TCN2d_inf receives the already sliced input (168:833) according to the script's input preparation
        # It does *not* perform slicing here.

        # x = self.inp_quant(x) # commented out in original

        for block in self.temporal_blocks:
            x = block(x)

        x = x.value if hasattr(x, 'value') else x # Get value from QuantTensor before FC
        x = self.fc(x)
        x = x.value if hasattr(x, 'value') else x # Get value from QuantTensor after FC
        # x = self.out_quant(x) # commented out in original
        # x = x.value if hasattr(x, 'value') else x # Get value from QuantTensor after out_quant

        # Final output reshape (assuming FC reduces spatial to 1x1)
        x = x.reshape(x.size(0), -1)

        return x

# --- Instantiate blocks (do this once) ---
# Ensure these match the model that generated your ONNX file
# Updated based on inferred logic for TCN2d_inf
block1_orig = TemporalBlock2d(1, 4, kernel_size=9, dilation=1,  stride=2,  dropout=0.05)
block2_orig = TemporalBlock2d(4, 8, kernel_size=9, dilation=4,  stride=1,  dropout=0.05)
block3_orig = TemporalBlock2d(8, 16, kernel_size=9, dilation=16, stride=1,  dropout=0.05)
custom_blocks_orig = [block1_orig, block2_orig, block3_orig]

# Ensure these match the inference model architecture that generated your ONNX file
# Based on TemporalBlock2d_inf __init__ calls
block1_inf_pt = TemporalBlock2d_inf(1, 4,  kernel_size=9,    stride=2,  dropout=0.05, use_stride = True, last_block=False) # conv1 stride 2, conv2 stride 2, no SliceSelector
block2_inf_pt = TemporalBlock2d_inf(4, 8,  kernel_size=9,    stride=1,  dropout=0.05, use_stride = False, last_block=True) # conv1 stride 1, conv2 stride 1, SliceSelector(4)
block3_inf_pt = TemporalBlock2d_inf(8, 16, kernel_size=9,    stride=1,  dropout=0.05, use_stride = False, last_block=False) # conv1 stride 1, conv2 stride 1, no SliceSelector
custom_blocks_inf_pt = [block1_inf_pt, block2_inf_pt, block3_inf_pt]


# ==============================================================================
# --- Intermediate Output Comparison Setup ---
# ==============================================================================

# Dictionary to store PyTorch intermediate outputs
intermediate_pt_outputs = {}

# List to store details of intermediate mismatches found
intermediate_mismatch_details = []

# Hook function to capture outputs
def save_intermediate_output(name):
    def hook(module, input, output):
        # For QuantTensor, get the value attribute (which is a standard tensor)
        output_tensor = output.value if hasattr(output, 'value') else output
        # Save a detached copy to avoid issues with graph building/memory
        # Add module type to name for uniqueness
        hook_name = f"{name}_{type(module).__name__}"
        intermediate_pt_outputs[hook_name] = output_tensor.detach()
    return hook

# Dictionary to store hook handles so they can be removed later
hook_handles = []

# --- Manual Mapping from PyTorch Module Name + Type to ONNX Output Tensor Name ---
# !!! IMPORTANT !!!
# YOU NEED TO CUSTOMIZE THIS DICTIONARY BASED ON YOUR SPECIFIC ONNX GRAPH.
# Use the `print_onnx_node_info` function below to inspect your ONNX model
# and find the names of the output tensors for the nodes corresponding to
# these PyTorch layers:
# - Each qnn.QuantConv2d layer's output
# - Each qnn.QuantReLU layer's output
# - Add Batchnorm outputs IF they are not fused in your ONNX
# - Add Slice/MaxPool/AvgPool outputs IF applicable and not fused
#
# The key should be the string name formed by combining the PyTorch module's
# `name` from `named_modules()` and its `type.__name__` (e.g., 'temporal_blocks.0.conv1_QuantConv2d').
# The value should be the exact name of the corresponding output tensor in the ONNX graph.
#
# To find ONNX names:
# 1. Set the ONNX_MODEL_PATH correctly above.
# 2. Uncomment the `print_onnx_node_info` call below and run the script once.
# 3. Look for nodes like 'QLinearConv', 'Conv', 'Relu', 'BatchNormalization', 'Slice', 'MaxPool', 'AveragePool', 'Add', 'Mul'. Their `output[0]` name is what you need.
#    Node names might also help identify the corresponding PyTorch layer (e.g., names containing 'temporal_blocks.0.conv1').
#
# Example ONNX tensor names might look like:
# - 'temporal_blocks.0/QuantConv2d[conv1]/Conv_output'
# - 'temporal_blocks.0/QuantConv2d[conv1]/Output'
# - 'temporal_blocks.0/QuantConv2d[conv1].output'
# - 'tcn_model/temporal_blocks.0/conv1.output'
# - 'tcn_model/temporal_blocks.0/QuantReLU[relu]/Relu_output'
# - 'tcn_model/temporal_blocks.0/relu.output'
# - 'tcn_model/fc.output'
#
# **You MUST fill this out correctly for the intermediate comparison to work.**
PYTORCH_TO_ONNX_TENSOR_MAP = {
    # Example mapping for the 'inference' model (adjust based on your ONNX file!)
    # THESE ARE PLACEHOLDERS. UPDATE THEM WITH THE ACTUAL TENSOR NAMES FROM YOUR ONNX.
    # You can get these names by uncommenting the print_onnx_node_info call below.

    # Temporal Block 0
    'temporal_blocks.0.conv1_QuantConv2d':                      'Conv_0_out0', # <--- REPLACE
    # If BatchNorm is fused, its output tensor name might not exist.
    'temporal_blocks.0.bn1_BatchNorm2d':                         None, # <--- Replace None or remove if fused
    'temporal_blocks.0.conv2_QuantConv2d':                      'Conv_1_out0', # <--- REPLACE
    'temporal_blocks.0.bn2_BatchNorm2d':                         None, # <--- Replace None or remove if fused

    # Temporal Block 1 (includes SliceSelectorStep)
    'temporal_blocks.1.conv1_QuantConv2d':                      'Conv_2_out0', # <--- REPLACE
    'temporal_blocks.1.bn1_BatchNorm2d':                         None, # <--- Replace None or remove if fused
    'temporal_blocks.1.conv2_QuantConv2d':                      'Conv_3_out0', # <--- REPLACE
    # Check if the SliceSelector output tensor has a distinct name in ONNX
    # It might be the input to the next node (bn2 or relu in this case based on code structure)
    'temporal_blocks.1.slice_selector_step_SliceSelectorStep':  'Slice_0_out0', # <--- REPLACE (if exists)
    'temporal_blocks.1.bn2_BatchNorm2d':                         None, # <--- Replace None or remove if fused

    # Temporal Block 2
    'temporal_blocks.2.conv1_QuantConv2d':                      'Conv_4_out0', # <--- REPLACE
    'temporal_blocks.2.bn1_BatchNorm2d':                         None, # <--- Replace None or remove if fused
    'temporal_blocks.2.conv2_QuantConv2d':                      'Conv_5_out0', # <--- REPLACE
    'temporal_blocks.2.bn2_BatchNorm2d':                         None, # <--- Replace None or remove if fused

    # Final FC Layer (often a Gemm or another Conv depending on the export)
    # Note: If there's a spatial reduction (like pooling or slicing) right before the FC in PT
    # and it corresponds to an ONNX node, you might want to map its output too.
    'fc_QuantConv2d':                                           'Conv_6_out0', # <--- REPLACE (or Gemm_X_out0 etc.)
}
# You would add a similar map for MODEL_CHOICE == "original" if needed.

# --- Helper to print ONNX node info for mapping ---
def print_onnx_node_info(onnx_model_path):
    """Loads an ONNX model and prints names and output tensor names of relevant nodes."""
    print(f"\n--- Inspecting ONNX Model: {onnx_model_path} ---")
    try:
        model = ModelWrapper(onnx_model_path)
        print(f"Input Name: {model.graph.input[0].name}")
        print(f"Output Name: {model.graph.output[0].name}")
        print("\nRelevant Nodes (Conv, QLinearConv, Relu, BatchNormalization, Slice, MaxPool, AveragePool, Add, Mul, etc.):")
        for i, node in enumerate(model.graph.node):
            # Check node name and op_type for potential layer outputs
            if node.op_type in ['Conv', 'QLinearConv', 'Relu', 'BatchNormalization', 'Slice', 'MaxPool', 'AveragePool', 'Add', 'Mul', 'Gemm', 'MultiThreshold']: # Added MultiThreshold
                 # Check if it has at least one output tensor
                if node.output:
                    print(f"  Node {i}:")
                    print(f"    Name: {node.name}")
                    print(f"    OpType: {node.op_type}")
                    print(f"    Output Tensor Name(s): {node.output}")

                    # print(f"    Input Tensor Name(s): {node.input}") # Uncomment for more detail
        print("--- Inspection Complete ---")
        print("Use the 'Output Tensor Name(s)' above to populate the PYTORCH_TO_ONNX_TENSOR_MAP dictionary.")
        print("Match PyTorch module names like 'temporal_blocks.0.conv1', 'temporal_blocks.0.relu', 'fc', etc.")
        print("Remember the dictionary key should be 'PyTorch_module_name_PyTorch_ModuleType' (e.g., 'temporal_blocks.0.conv1_QuantConv2d').")
        print("If a PyTorch layer is fused or removed in ONNX (like often happens with BatchNorm), map its key to None or remove it from the map.")


    except FileNotFoundError:
        print(f"ERROR: ONNX model file not found at {onnx_model_path}")
    except Exception as e:
        print(f"ERROR inspecting ONNX model: {e}")

# Uncomment the line below to run the inspection and help build the map:
#print_onnx_node_info(ONNX_MODEL_PATH)
#sys.exit("ONNX inspection finished. Update PYTORCH_TO_ONNX_TENSOR_MAP and re-run.") # Exit after inspection


# ==============================================================================

# --- Load Data ---
print("Loading data from .pt file...")
inputs = None
try:
    # Ensure weights_only=False is used if data isn't stored as weights
    # map_location='cpu' is good practice for loading on any machine
    data_dict = torch.load(INPUT_DATA_PATH, map_location=torch.device('cpu'), weights_only=False)
    if isinstance(data_dict, dict):
        input_key = 'inputs' # Assumes key is 'inputs'
        if input_key not in data_dict:
            # Find a tensor-like object if 'inputs' isn't the key
            possible_keys = [k for k, v in data_dict.items() if isinstance(v, (torch.Tensor, np.ndarray))]
            if not possible_keys:
                 raise KeyError(f"Input key '{input_key}' not found and no other tensor-like objects found.")
            input_key = possible_keys[0]
            print(f"Warning: Input key '{input_key}' not found. Using first found tensor/ndarray key: '{input_key}'")

        inputs_tensor = data_dict[input_key]
        if isinstance(inputs_tensor, torch.Tensor):
            # Convert to numpy and ensure float32 as ONNX expects
            inputs = inputs_tensor.detach().numpy().astype(np.float32)
        elif isinstance(inputs_tensor, np.ndarray):
            inputs = inputs_tensor.astype(np.float32)
        else:
            raise TypeError(f"Loaded input data is not Tensor/ndarray: {type(inputs_tensor)}")

        # Assuming input shape should be N x 1 x Length x 1 (NCHW)
        # If your loaded data is N x Length, reshape it.
        if inputs.ndim == 2 and inputs.shape[1] == 1000:
             inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1], 1)
             print(f"Reshaped input data from (N, L) to (N, 1, L, 1): {inputs.shape}")
        elif inputs.ndim == 4 and inputs.shape[1] == 1 and inputs.shape[3] == 1 and inputs.shape[2] == 1000:
             print(f"Loaded input data shape (N, 1, 1000, 1): {inputs.shape}")
        else:
             print(f"Warning: Input data shape {inputs.shape} is not the expected (N, 1, 1000, 1). Proceeding but verify.")


    else:
         raise TypeError(f"Unexpected data type loaded from {INPUT_DATA_PATH}: {type(data_dict)}. Expected a dictionary.")
except FileNotFoundError:
    print(f"ERROR: Input data file not found at {INPUT_DATA_PATH}")
except KeyError as e:
    print(f"ERROR: Could not find expected input data in the loaded .pt file: {e}")
except Exception as e:
    print(f"ERROR loading or processing input data: {e}")

# --- Load Selected PyTorch Model ---
pytorch_model = None
print(f"\nLoading PyTorch model structure: {MODEL_CHOICE}...")
try:
    if inputs is None: raise ValueError("Input data not loaded.") # Check if data loading succeeded

    if MODEL_CHOICE == "original":
        pytorch_model = TCN2d(custom_blocks=custom_blocks_orig, num_outputs=NUM_OUTPUT_CLASSES)
        # The original PyTorch model forward pass is assumed to handle the full input
        apply_input_slice_to_onnx = False # Feed full 1000 length to ONNX for original model
    elif MODEL_CHOICE == "inference":
        pytorch_model = TCN2d_inf(custom_blocks=custom_blocks_inf_pt, num_outputs=NUM_OUTPUT_CLASSES)
        # The ONNX for the inference model is assumed to expect the sliced input
        # The PyTorch TCN2d_inf model is assumed to take the sliced input directly.
        apply_input_slice_to_onnx = True # Apply slice 168:833 *before* feeding to ONNX
    else:
        raise ValueError("MODEL_CHOICE must be 'original' or 'inference'")

    # Load weights
    checkpoint = torch.load(PYTORCH_WEIGHTS_PATH, map_location=torch.device('cpu'), weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint) # Handle checkpoints that might wrap the state_dict

    # Load weights, potentially ignoring missing/unexpected keys if architectures differ slightly
    # For strict matching, remove strict=False
    missing_keys, unexpected_keys = pytorch_model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        print(f"Warning: Unexpected keys found in checkpoint for {MODEL_CHOICE} model: {unexpected_keys}")
    if missing_keys:
        print(f"Warning: Missing keys in {MODEL_CHOICE} model for checkpoint: {missing_keys}")

    pytorch_model.eval() # Set model to evaluation mode (disables dropout, etc.)
    print(f"PyTorch model '{MODEL_CHOICE}' loaded.")
except FileNotFoundError:
    print(f"ERROR: PyTorch weights file not found: {PYTORCH_WEIGHTS_PATH}")
    pytorch_model = None
except Exception as e:
    print(f"ERROR loading PyTorch model or weights: {e}")
    pytorch_model = None

# --- Load ONNX Model ---
print(f"\nLoading ONNX model: {ONNX_MODEL_PATH}...")
model_for_sim = None
input_name_onnx = None
output_name_onnx = None
try:
    if not os.path.exists(ONNX_MODEL_PATH):
         raise FileNotFoundError(f"ONNX model file not found at {ONNX_MODEL_PATH}")
    model_for_sim = ModelWrapper(ONNX_MODEL_PATH)
    input_name_onnx = model_for_sim.graph.input[0].name
    output_name_onnx = model_for_sim.graph.output[0].name
    print(f"ONNX Input Name: {input_name_onnx}")
    print(f"ONNX Output Name: {output_name_onnx}")
    print("ONNX model loaded.")
except FileNotFoundError as e:
     print(f"ERROR: {e}")
except Exception as e:
    print(f"ERROR loading ONNX model: {e}")


# --- Attach Hooks to PyTorch Model ---
if pytorch_model is not None:
    print("\nAttaching hooks to PyTorch model...")
    # Iterate through all named modules in the model
    for name, module in pytorch_model.named_modules():
        # Attach hooks to relevant layer types for intermediate comparison
        # Include Batchnorm and SliceSelectorStep if you map their outputs
        # Check if the constructed hook name exists in the mapping before attaching (more selective)
        hook_name = f"{name}_{type(module).__name__}"
        # print(f"  - Checking hook for {hook_name} ({type(module).__name__})")
        # Only attach hook if this layer is in our map AND its ONNX counterpart is not None
        # This prevents hooking layers whose ONNX output is fused/missing
        if hook_name in PYTORCH_TO_ONNX_TENSOR_MAP and PYTORCH_TO_ONNX_TENSOR_MAP[hook_name] is not None:
             # Skip the top-level TCN2d/TCN2d_inf module itself if it somehow matches
             if name == '':
                 continue
             print(f"  - Attaching hook to {hook_name}")
             # Use hook_name directly as the key for the dictionary
             handle = module.register_forward_hook(save_intermediate_output(hook_name))
             hook_handles.append(handle)
        # else:
        #      print(f"  - Skipping hook for {name} ({type(module).__name__}) - Not in map or ONNX counterpart is None.")

    if not hook_handles and any(v is not None for v in PYTORCH_TO_ONNX_TENSOR_MAP.values()):
         print("WARNING: No hooks were attached, but PYTORCH_TO_ONNX_TENSOR_MAP is not empty (and has non-None values). Check if module names and types match the model's named modules.")
    elif not hook_handles and not any(v is not None for v in PYTORCH_TO_ONNX_TENSOR_MAP.values()) and PYTORCH_TO_ONNX_TENSOR_MAP:
        print("Info: No hooks attached because all entries in PYTORCH_TO_ONNX_TENSOR_MAP are None.")


    print("Hooks attached.")

# ==============================================================================
# --- Comparison Loop ---
# ==============================================================================
print("\n--- Starting Comparison ---")
total_final_mismatches = 0
intermediate_mismatch_details = [] # Reset before loop
samples_compared = 0
samples_where_comparison_attempted = 0 # New counter for samples where comparison was tried

# Check if everything loaded correctly before starting
can_compare = pytorch_model is not None and inputs is not None and model_for_sim is not None

if not PYTORCH_TO_ONNX_TENSOR_MAP and can_compare:
    print("WARNING: PYTORCH_TO_ONNX_TENSOR_MAP is empty. Intermediate output comparison will be skipped entirely.")


if can_compare:
    num_to_run = min(NUM_SAMPLES_TO_TEST, inputs.shape[0])
    if num_to_run == 0:
        print("ERROR: No input samples found to run comparison.")
    else:
        for i in range(num_to_run):
            print(f"\n--- Sample {i+1}/{num_to_run} ---")
            samples_compared += 1 # Increment here

            # 1. Prepare Input for PyTorch
            pytorch_input_tensor = None
            pt_execution_successful = False
            try:
                # PyTorch model receives the input according to its architecture.
                current_sample_full = inputs[i:i+1] # Keep batch dimension (1, 1, 1000, 1)

                if MODEL_CHOICE == "inference":
                    # TCN2d_inf expects sliced input
                    pytorch_input_np = current_sample_full[:, :, INPUT_SLICE_START:INPUT_SLICE_END, :]
                    print(f"  PyTorch Input Shape (Sliced for Inference Model): {pytorch_input_np.shape}")
                else: # MODEL_CHOICE == "original"
                     # TCN2d_orig expects full input
                    pytorch_input_np = current_sample_full
                    print(f"  PyTorch Input Shape (Full for Original Model): {pytorch_input_np.shape}")

                pytorch_input_tensor = torch.from_numpy(pytorch_input_np).float()
                pt_execution_successful = True # Input prepared
            except Exception as e:
                print(f"ERROR preparing PyTorch input for sample {i+1}: {e}")
                pt_execution_successful = False
                # Mismatch count handled later if execution failed

            # 2. Run PyTorch Inference (capturing intermediates via hooks)
            pytorch_output_np = None # Initialize
            if pt_execution_successful:
                try:
                    intermediate_pt_outputs.clear() # Clear dictionary before each run
                    with torch.no_grad():
                        # The forward pass will trigger the hooks and populate intermediate_pt_outputs
                        pytorch_output_tensor = pytorch_model(pytorch_input_tensor)
                    pytorch_output_np = pytorch_output_tensor.detach().numpy()
                    # print(f"  PyTorch Final Output Shape: {pytorch_output_np.shape}")
                    pt_execution_successful = True # Execution completed
                except Exception as e:
                    print(f"ERROR running PyTorch inference ({MODEL_CHOICE} model) for sample {i+1}: {e}")
                    pt_execution_successful = False
                    # Mismatch count handled later if execution failed

            # 3. Prepare Input for ONNX Simulation
            onnx_input_dict = None
            onnx_input_np = None
            onnx_input_prepared = False
            if pt_execution_successful: # Only prepare ONNX input if PT input worked
                try:
                    # The input to ONNX depends on MODEL_CHOICE and the slicing config
                    if apply_input_slice_to_onnx:
                        # ONNX expects sliced input for the inference model
                        onnx_input_np = current_sample_full[:, :, INPUT_SLICE_START:INPUT_SLICE_END, :]
                        print(f"  ONNX Input Shape (Sliced): {onnx_input_np.shape}")
                    else:
                        # ONNX expects full input for the original model
                        onnx_input_np = current_sample_full
                        print(f"  ONNX Input Shape (Full): {onnx_input_np.shape}")

                    # execute_onnx needs float32 input dict
                    onnx_input_dict = {input_name_onnx: onnx_input_np.astype(np.float32)}
                    onnx_input_prepared = True
                except Exception as e:
                    print(f"ERROR preparing ONNX input for sample {i+1}: {e}")
                    onnx_input_prepared = False
                    # Mismatch count handled later if execution failed

            # 4. Run ONNX Inference (using execute_onnx and capturing intermediates)
            onnx_output_np = None
            onnx_intermediates = {}
            onnx_execution_successful = False
            if onnx_input_prepared: # Only run ONNX if its input worked
                try:
                    # Setting return_full_exec_context=True gives access to intermediate tensors
                    # This returns a dictionary where keys are tensor names and values are numpy arrays
                    full_context = execute_onnx(model_for_sim, onnx_input_dict, return_full_exec_context=True)

                    # Check if the top-level 'outputs' key exists
                    if 'outputs' in full_context:
                        output_dict = full_context['outputs']
                        # Now check if the specific output tensor name exists within the outputs dictionary
                        if output_name_onnx in output_dict:
                            onnx_output_np = output_dict[output_name_onnx] # Correct access!
                            onnx_execution_successful = True # Mark as successful if final output found
                            # print(f"  ONNX Sim Final Output Shape: {onnx_output_np.shape}")
                        else:
                            # The 'outputs' key exists, but the specific final output tensor name wasn't found
                            print(f"    ERROR: Final output tensor '{output_name_onnx}' not found in ONNX outputs dict for sample {i+1}.")
                    else:
                        # The top-level 'outputs' key itself was not found
                        print(f"    ERROR: 'outputs' key not found in execute_onnx result for sample {i+1}.")


                    # intermediate_outputs check remains the same
                    if 'intermediate_outputs' in full_context:
                        onnx_intermediates = full_context['intermediate_outputs']
                    else:
                        print(f"    WARNING: 'intermediate_outputs' key not found in execute_onnx result for sample {i+1}.")


                except Exception as e:
                    print(f"ERROR running ONNX simulation (execute_onnx) for sample {i+1}: {e}")
                    if onnx_input_dict and input_name_onnx in onnx_input_dict:
                         print(f"  Input dictionary causing error: {{'{input_name_onnx}': shape {onnx_input_dict[input_name_onnx].shape}, dtype {onnx_input_dict[input_name_onnx].dtype}}}")
                    else:
                         print("  ONNX Input dictionary could not be prepared or was None.")
                    onnx_execution_successful = False


            # --- Comparison Logic ---

            # Determine if comparison can be attempted for this sample
            # Final comparison needs both PT output and ONNX final output
            # Intermediate comparison needs PT outputs (from hooks) and ONNX intermediates
            can_do_final_comparison = pt_execution_successful and onnx_execution_successful and pytorch_output_np is not None and onnx_output_np is not None
            # Intermediate comparison also needs the map to not be empty AND some intermediates to be available
            can_do_intermediate_comparison = pt_execution_successful and onnx_execution_successful and bool(PYTORCH_TO_ONNX_TENSOR_MAP) and bool(onnx_intermediates)

            if can_do_final_comparison or can_do_intermediate_comparison:
                 samples_where_comparison_attempted += 1 # Count this sample as attempted

            # 5. Compare Final Outputs
            if can_do_final_comparison:
                 print("  Comparing Final Outputs:")
                 if pytorch_output_np.shape != onnx_output_np.shape:
                     print(f"    ERROR: Shape mismatch! PyTorch={pytorch_output_np.shape}, ONNX={onnx_output_np.shape}")
                     total_final_mismatches += 1
                 else:
                     try:
                         onnx_output_float = onnx_output_np.astype(np.float32)
                         pytorch_output_float = pytorch_output_np.astype(np.float32)

                         if not np.allclose(pytorch_output_float, onnx_output_float, rtol=RTOL, atol=ATOL):
                             print(f"    WARNING: Numerical mismatch detected for FINAL output!")
                             total_final_mismatches += 1
                             abs_diff = np.abs(pytorch_output_float - onnx_output_float)
                             rel_diff = abs_diff / (np.maximum(np.abs(pytorch_output_float), np.abs(onnx_output_float)) + 1e-9)
                             current_max_abs = abs_diff.max()
                             current_max_rel = rel_diff.max()
                             print(f"      Max Absolute Difference: {current_max_abs:.6f}")
                             print(f"      Max Relative Difference: {current_max_rel:.6f}")
                         # else:
                         #    print("    Final outputs match numerically (within tolerance).")
                     except Exception as e:
                         print(f"    ERROR during final output comparison for sample {i+1}: {e}")
                         total_final_mismatches += 1 # Count comparison error as a mismatch
            elif not (pt_execution_successful and onnx_input_prepared and onnx_execution_successful):
                 # Only count as final mismatch if executions didn't complete successfully (implying no comparison possible)
                 # We already printed an error message about why execution failed.
                 pass # Don't increment total_final_mismatches here, it's already implicitly a failure case handled by lack of comparison
            else:
                 # This case means can_do_final_comparison is False, but executions *were* successful?
                 # This shouldn't happen with the current logic, but as a fallback
                 print(f"  Final output comparison skipped for sample {i+1} due to unexpected state.")


            # 6. Compare Intermediate Outputs
            if can_do_intermediate_comparison:
                 print("  Comparing Intermediate Outputs:")
                 layer_mismatches_this_sample = 0
                 compared_layers = 0

                 for pt_name_hook, onnx_tensor_name in PYTORCH_TO_ONNX_TENSOR_MAP.items():
                     # Skip comparison for layers mapped to None
                     if onnx_tensor_name is None:
                         continue

                     compared_layers += 1 # Only count layers that are mapped and not None

                     pt_output = intermediate_pt_outputs.get(pt_name_hook)
                     onnx_output = onnx_intermediates.get(onnx_tensor_name)

                     if pt_output is None:
                         print(f"    WARNING: PyTorch intermediate output for '{pt_name_hook}' not captured by hook or name mismatch.")
                         # Only count as mismatch if ONNX counterpart exists
                         if onnx_output is not None:
                              layer_mismatches_this_sample += 1
                              intermediate_mismatch_details.append(f"Sample {i+1}: PyTorch output missing for '{pt_name_hook}' (ONNX tensor '{onnx_tensor_name}')")
                         continue

                     if onnx_output is None:
                         print(f"    WARNING: ONNX intermediate output tensor '{onnx_tensor_name}' not found in ONNX intermediates (mapped from PT '{pt_name_hook}').")
                         layer_mismatches_this_sample += 1 # Count if PT output exists but ONNX doesn't
                         intermediate_mismatch_details.append(f"Sample {i+1}: ONNX output missing for '{onnx_tensor_name}' (Mapped from PT '{pt_name_hook}')")
                         continue

                     # Ensure numpy arrays and float32 for comparison
                     pt_output_np = pt_output.numpy().astype(np.float32)
                     onnx_output_np = onnx_output.astype(np.float32)

                     if pt_output_np.shape != onnx_output_np.shape:
                         print(f"    ERROR: Shape mismatch for layer '{pt_name_hook}' (ONNX tensor '{onnx_tensor_name}')! PyTorch={pt_output_np.shape}, ONNX={onnx_output_np.shape}")
                         layer_mismatches_this_sample += 1
                         intermediate_mismatch_details.append(f"Sample {i+1}: Shape mismatch for '{pt_name_hook}' (ONNX '{onnx_tensor_name}') PT:{pt_output_np.shape} vs ONNX:{onnx_output_np.shape}")
                     else:
                          try:
                             if not np.allclose(pt_output_np, onnx_output_np, rtol=RTOL, atol=ATOL):
                                 print(f"    WARNING: Numerical mismatch detected for layer '{pt_name_hook}' (ONNX tensor '{onnx_tensor_name}')!")
                                 layer_mismatches_this_sample += 1
                                 abs_diff = np.abs(pt_output_np - onnx_output_np)
                                 rel_diff = abs_diff / (np.maximum(np.abs(pt_output_np), np.abs(onnx_output_np)) + 1e-9)
                                 current_max_abs = abs_diff.max()
                                 current_max_rel = rel_diff.max()
                                 print(f"      Max Absolute Difference: {current_max_abs:.6f}")
                                 print(f"      Max Relative Difference: {current_max_rel:.6f}")
                                 intermediate_mismatch_details.append(f"Sample {i+1}: Mismatch for '{pt_name_hook}' (ONNX '{onnx_tensor_name}'). Max Abs Diff: {current_max_abs:.6f}, Max Rel Diff: {current_max_rel:.6f}")
                              # else:
                              #    print(f"    Layer '{pt_name_hook}' (ONNX tensor '{onnx_tensor_name}') outputs match (within tolerance).")
                          except Exception as e:
                             print(f"    ERROR during comparison for layer '{pt_name_hook}' (ONNX tensor '{onnx_tensor_name}'): {e}")
                             layer_mismatches_this_sample += 1
                             intermediate_mismatch_details.append(f"Sample {i+1}: Comparison ERROR for '{pt_name_hook}' (ONNX '{onnx_tensor_name}'): {e}")


                 num_mapped_and_non_none_layers = sum(1 for v in PYTORCH_TO_ONNX_TENSOR_MAP.values() if v is not None)
                 if num_mapped_and_non_none_layers > 0: # Only print summary if comparison was attempted for any mapped layer
                     if layer_mismatches_this_sample == 0:
                          print(f"    All {num_mapped_and_non_none_layers} configured intermediate layers match for this sample.")
                     else:
                          print(f"    Mismatches found in {layer_mismatches_this_sample}/{num_mapped_and_non_none_layers} configured intermediate layers for this sample.")
                 elif PYTORCH_TO_ONNX_TENSOR_MAP: # If map is not empty but all mapped values are None
                     print("    Intermediate comparison skipped: All mapped layers have None ONNX tensor names.")
                 # else if PYTORCH_TO_ONNX_TENSOR_MAP is empty, nothing is printed here.

            elif PYTORCH_TO_ONNX_TENSOR_MAP: # Comparison couldn't be attempted at all due to execution errors
                 print(f"  Intermediate comparison skipped for sample {i+1} due to execution errors (ONNX intermediates likely not available).")
            # If PYTORCH_TO_ONNX_TENSOR_MAP is empty, nothing is printed here, handled in final summary.


else:
    print("Prerequisites not met (check model/data/map loading). Skipping comparison loop.")


# --- Clean up hooks ---
print("\nRemoving PyTorch hooks...")
for handle in hook_handles:
    handle.remove()
print("Hooks removed.")


# --- Final Summary ---
print("\n--- Comparison Summary ---")
if not can_compare and (pytorch_model is None or inputs is None or model_for_sim is None):
     print("ERROR: Cannot perform comparison due to missing PyTorch model, input data, or ONNX model.")
elif samples_compared == 0:
     print("ERROR: No samples were successfully processed for comparison.")
else:
    print(f"Compared {samples_compared} sample(s).")
    # Final output summary based on samples where comparison was attempted vs. where it failed
    print(f"FINAL OUTPUT Summary:")
    # Count samples where final comparison was attempted and succeeded (0 mismatches)
    final_matches = samples_where_comparison_attempted - total_final_mismatches
    if samples_where_comparison_attempted > 0:
        if total_final_mismatches == 0:
             print(f"✅ All {samples_where_comparison_attempted} samples where comparison was attempted produced final outputs that match (within tolerance).")
        else:
             print(f"❌ Mismatches or comparison errors found in {total_final_mismatches}/{samples_where_comparison_attempted} sample(s) where comparison was attempted.")
    elif samples_compared > 0:
         print(f"  Final output comparison skipped for all {samples_compared} samples due to execution errors.")
    else: # samples_compared == 0 or other unexpected state
         print("  Final output comparison could not be completed.")


    num_mapped_layers = len(PYTORCH_TO_ONNX_TENSOR_MAP)
    num_mapped_and_non_none_layers = sum(1 for v in PYTORCH_TO_ONNX_TENSOR_MAP.values() if v is not None)

    total_intermediate_mismatches = len(intermediate_mismatch_details)
    total_possible_intermediate_comparisons_across_attempted_samples = samples_where_comparison_attempted * num_mapped_and_non_none_layers # Number of layers checked * total attempted samples

    print(f"INTERMEDIATE OUTPUTS Summary (checking {num_mapped_and_non_none_layers} layer outputs per sample where comparison was attempted):")

    if num_mapped_layers == 0:
        print("  No layers configured in PYTORCH_TO_ONNX_TENSOR_MAP for intermediate comparison.")
    elif num_mapped_and_non_none_layers == 0:
        print("  No layers configured in PYTORCH_TO_ONNX_TENSOR_MAP with non-None ONNX tensor names.")
    elif samples_where_comparison_attempted == 0:
         print(f"  Intermediate comparison was skipped for all {samples_compared} samples due to execution errors.")
    elif total_intermediate_mismatches == 0:
        print(f"  ✅ All {total_possible_intermediate_comparisons_across_attempted_samples} intermediate layer comparisons across {samples_where_comparison_attempted} samples match.")
    else:
        print(f"  ❌ Mismatches found in {total_intermediate_mismatches}/{total_possible_intermediate_comparisons_across_attempted_samples} intermediate layer comparisons across {samples_where_comparison_attempted} samples where comparison was attempted.")
        print("  --- Intermediate Mismatch Details (Sample: Layer Name (ONNX Tensor Name), Difference) ---")
        # Print the collected details
        for detail in intermediate_mismatch_details:
            print(f"  - {detail}")
        print("  ---------------------------------------------------------------------------------------")


    print(f"   Tolerance (ATOL, RTOL): ({ATOL}, {RTOL})")

print("--------------------------")

#--- END OF FILE compare_onnx_inference_to_torch.py ---