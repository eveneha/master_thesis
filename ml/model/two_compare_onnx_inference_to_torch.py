# --- Imports ---
import os
import sys
import json
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt # Keep commented unless plotting
import random
import onnx # Make sure onnx is imported

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
from torchinfo import summary
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat, Int8WeightPerChannelFloat, Uint8ActPerTensorFloat

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_shapes import InferShapes # Needed?
# from qonnx.transformation.general import SortGraph # Needed?


import logging
# Set higher logging level to reduce verbose QONNX messages unless debugging quantization
logging.getLogger("qonnx.core.datatype").setLevel(logging.WARNING)
logging.getLogger("qonnx.util.basic").setLevel(logging.WARNING)

# ==============================================================================
# --- Configuration - VERIFY THESE PATHS AND FACTORS ---
# ==============================================================================

MODEL_CHOICE = "original" # Should match the PyTorch class used
# --- ONNX model AFTER specialization, containing HLS/RTL nodes ---
ONNX_MODEL_PATH = "/home/eveneiha/finn/workspace/finn/onnx/21_tcn_before_specialize.onnx"
PYTORCH_WEIGHTS_PATH = "/home/eveneiha/finn/workspace/ml/model/tcn_model_v41_avg_pool.pth" # Weights matching TCN2d_inf
INPUT_DATA_PATH = "/home/eveneiha/finn/workspace/ml/data/preprocessed/test.pt" # Contains full 1000 len signals

# --- Quantization / Dequantization ---
# TODO: VERIFY THESE! Extract from the ONNX graph if possible.
# Assuming input is INT8 without scaling for now (simplest case)
INPUT_QUANTIZATION_ENABLED = True # Set to True to quantize input for ONNX
# Input scale factor (if needed for quantize_to_finn_datatype) - Placeholder
# INPUT_SCALE_FACTOR = 1.0
# Combined scale factor for final output dequantization
OUTPUT_DEQUANT_SCALE_FACTOR = 0.007948528975248337 # Needs verification

def quantize_input(x_float, scale=OUTPUT_DEQUANT_SCALE_FACTOR, zero_point=0):
                            x_int = np.clip(np.round(x_float / scale) + zero_point, -128, 127)
                            return x_int.astype(np.int8)
# --- Before the loop ---
# Define the scale factors directly from the values you extracted
# Shape (1, 5, 1, 1) initially based on the nested list
scale_factors_raw = np.array([
    [[[0.00017027670401148498]]],
    [[[0.0001750222290866077]]],
    [[[0.00016915365995373577]]],
    [[[0.00022898473253007978]]],
    [[[0.00019804797193501145]]]
], dtype=np.float32)

# Reshape for broadcasting with the flattened (1, 5) accumulator/output
scale_factors_reshaped = scale_factors_raw.reshape(1, 5)
print(f"Using extracted scale factors: shape {scale_factors_reshaped.shape}")
NUM_SAMPLES_TO_TEST = 1
NUM_OUTPUT_CLASSES = 5

# Tolerances for float comparison after dequantization
ATOL = 1e-6 # Absolute tolerance might need adjustment
RTOL = 1e-5 # Relative tolerance might need adjustment

print(f"--- Configuration ---")
print(f"MODEL_CHOICE: {MODEL_CHOICE}")
print(f"PyTorch Weights: {PYTORCH_WEIGHTS_PATH}")
print(f"Input Data: {INPUT_DATA_PATH}")
print(f"Target ONNX Model: {ONNX_MODEL_PATH}")
print(f"Input Quantization Enabled: {INPUT_QUANTIZATION_ENABLED}")
print(f"Output Dequantization Scale Factor: {OUTPUT_DEQUANT_SCALE_FACTOR:.8f}")
print(f"ATOL: {ATOL}, RTOL: {RTOL}")
print(f"--------------------")

# ==============================================================================
# --- PyTorch Model Definitions ---
# ==============================================================================

# --- Redefinition of TCN to CNN for inference ---   

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

        # if slice_name is not None:
        #     dilation = 1
            
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
        #self.slice_selector = SliceSelector(slice_name) 
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        #x = self.slice_selector(x) # only on for the final block 
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

        # self.reduce = qnn.QuantConv2d(
        #     in_channels=last_out_channels,
        #     out_channels=last_out_channels,
        #     kernel_size=(168, 1),  # or use known dims if static
        #     weight_quant=Int8WeightPerChannelFloat,
        #     input_quant=Int8ActPerTensorFloat,
        #     weight_bit_width=8,
        #     act_bit_width=8,
        #     bias=False
        # )

        
        # Input quant layer
        self.inp_quant = qnn.QuantIdentity(
            bit_width=8,
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        self.out_quant = qnn.QuantIdentity(
            bit_width=8,
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )


    def forward(self, x): # Input is float
        #x = self.inp_quant(x) 
        for block in self.temporal_blocks:
            x = block(x) 
        #x = self.reduce(x)# Pass INT8 tensor value
        x = x[:, :, 84:85, :]  # 0-based index
        x = self.fc(x) # Output likely INT32 internal or QuantTensor
        #x_val = self.out_quant(x_val)  # Final quantization to INT8
        x_val = x.value if hasattr(x, 'value') else x
        x_reshaped = x_val.reshape(x_val.size(0), -1)
        return x_reshaped
    
# --- Instantiate blocks (do this once) ---
block1_orig = TemporalBlock2d(1, 4, kernel_size=9, dilation=1,  stride=2,  dropout=0.05)
block2_orig = TemporalBlock2d(4, 8, kernel_size=9, dilation=4,  stride=1,  dropout=0.05)
block3_orig = TemporalBlock2d(8, 16, kernel_size=9, dilation=16, stride=1,  dropout=0.05)
custom_blocks_orig = [block1_orig, block2_orig, block3_orig]

class SliceSelectorStep(nn.Module):
    def __init__(self, step = 1):
        super().__init__()
        self.step = step
               
    def forward(self, x):
        #print(f"Step size: {self.step}")
        return x[:, :, :65:self.step, :]  

class TemporalBlock2d_inf(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dropout=0.2, use_stride = False, last_block = False):
        super(TemporalBlock2d_inf, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.last_block = last_block
        
        if last_block:
            self.slice_selector_step = SliceSelectorStep(4)
            
        else: 
            self.slice_selector_step = None
        
        
        self.conv1 = qnn.QuantConv2d(
            n_inputs, n_outputs,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(0, 0),  # no padding – only compute valid outputs
            #dilation=(dilation, 1),
            weight_quant=Int8WeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat,
            weight_bit_width=8,
            act_bit_width=8,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)
         
        if use_stride: 
            stride = 4
        else:
            stride = 1
               
        self.conv2 = qnn.QuantConv2d(
            n_outputs, n_outputs,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(0, 0),
            #dilation=(dilation, 1),
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
                


        # self.out_quant = qnn.QuantIdentity(
        #     bit_width=8,
        #     act_quant=Int8ActPerTensorFloat,
        #     return_quant_tensor=True
        # )
        

    def forward(self, x):
        #x = x[:, :, 168:833, :]  # 0-based index
        for block in self.temporal_blocks:
            x = block(x)
            
        x = self.fc(x)
        #x = self.out_quant(x)
        x = x.reshape(x.size(0), -1)
        return x
    
    

# print(pruning_map["block3_conv2"])
# print(pruning_map["block2_conv2"])
block1_inf = TemporalBlock2d_inf(1, 4,  kernel_size=9, stride=2, dropout=0.05, use_stride = True)
block2_inf = TemporalBlock2d_inf(4, 8,  kernel_size=9, stride=1, dropout=0.05, last_block=True)
block3_inf = TemporalBlock2d_inf(8, 16, kernel_size=9, stride=1, dropout=0.05)
custom_blocks_inf_pt = [block1_inf, block2_inf, block3_inf]
# ==============================================================================

# --- Load Data ---
print("\nLoading data from .pt file...")
inputs = None
try:
    if not os.path.exists(INPUT_DATA_PATH): raise FileNotFoundError(f"Input data file not found at {INPUT_DATA_PATH}")
    data_dict = torch.load(INPUT_DATA_PATH, map_location=torch.device('cpu'), weights_only=False)
    if isinstance(data_dict, dict):
        input_key = 'inputs';
        if input_key not in data_dict: raise KeyError(f"Input key '{input_key}' not found.")
        inputs_tensor = data_dict[input_key]
        if isinstance(inputs_tensor, torch.Tensor): inputs = inputs_tensor.detach().numpy().astype(np.float32)
        elif isinstance(inputs_tensor, np.ndarray): inputs = inputs_tensor.astype(np.float32)
        else: raise TypeError(f"Loaded data type error: {type(inputs_tensor)}")
        print(f"Loaded input data shape: {inputs.shape}") # Should be (N, 1, 1000, 1) or similar
        #print(inputs[0])
    else: raise TypeError(f"Unexpected loaded data type: {type(data_dict)}")
except Exception as e: print(f"ERROR loading data: {e}"); inputs = None

# --- Load PyTorch Model ---
pytorch_model = None
print(f"\nLoading PyTorch model structure: {MODEL_CHOICE}...")
try:
    if inputs is None: raise ValueError("Cannot load PyTorch model without input data.")
    if MODEL_CHOICE == "inference":
        pytorch_model = TCN2d_inf(custom_blocks=custom_blocks_inf_pt, num_outputs=NUM_OUTPUT_CLASSES)
        # IMPORTANT: Make sure the forward method of TCN2d_inf does NOT do internal slicing
    elif MODEL_CHOICE == "original":
        pytorch_model = TCN2d(custom_blocks=custom_blocks_orig, num_outputs=NUM_OUTPUT_CLASSES)

    else:
        raise ValueError("MODEL_CHOICE must be 'inference' for this script setup")
    if not os.path.exists(PYTORCH_WEIGHTS_PATH): raise FileNotFoundError(f"Weights not found: {PYTORCH_WEIGHTS_PATH}")
    checkpoint = torch.load(PYTORCH_WEIGHTS_PATH, map_location=torch.device('cpu'), weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    missing_keys, unexpected_keys = pytorch_model.load_state_dict(state_dict, strict=False)
    if unexpected_keys: print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
    if missing_keys: print(f"Warning: Missing keys in model: {missing_keys}")
    pytorch_model.eval()
    print(f"PyTorch model '{MODEL_CHOICE}' loaded successfully.")
except Exception as e: print(f"ERROR loading PyTorch model: {e}"); pytorch_model = None

# --- Load ONNX Dataflow Model ---
dataflow_model = None
dataflow_input_name = None
dataflow_output_name = None
dataflow_input_dtype = None
dataflow_output_dtype = None

print(f"\nLoading ONNX dataflow model: {ONNX_MODEL_PATH}...")
try:
    if not os.path.exists(ONNX_MODEL_PATH): raise FileNotFoundError(f"ONNX dataflow model not found at {ONNX_MODEL_PATH}")
    dataflow_model = ModelWrapper(ONNX_MODEL_PATH)
    dataflow_input_name = dataflow_model.graph.input[0].name
    dataflow_output_name = dataflow_model.graph.output[0].name
    # Get expected dtypes from the graph
    dataflow_input_dtype = dataflow_model.get_tensor_datatype(dataflow_input_name)
    dataflow_output_dtype = dataflow_model.get_tensor_datatype(dataflow_output_name)

    print(f"Dataflow Input Name: {dataflow_input_name}")
    print(f"Dataflow Expected Input Type: {dataflow_input_dtype}") # Should be INT8 for 21_...
    print(f"Dataflow Output Name: {dataflow_output_name}")
    print(f"Dataflow Expected Output Type: {dataflow_output_dtype}") # Should be INT type (e.g., INT18, INT32)

    # Assertion to catch mismatch early
    assert str(dataflow_input_dtype).upper().startswith("INT"), \
        f"Dataflow model input '{dataflow_input_name}' is {dataflow_input_dtype}, expected an INT type for this script setup."

    print("ONNX dataflow model loaded successfully.")
except AssertionError as e: print(f"ERROR: {e}"); dataflow_model = None
except Exception as e: print(f"ERROR loading/setting ONNX dataflow model: {e}"); dataflow_model = None

# --- Comparison Loop ---
print("\n--- Starting Comparison ---")
max_abs_diff = 0.0; max_rel_diff = 0.0; mismatched_samples = 0; samples_compared = 0
can_compare = pytorch_model is not None and inputs is not None and dataflow_model is not None

if not can_compare:
    if pytorch_model is None: print("ERROR: PyTorch model failed to load.")
    if inputs is None: print("ERROR: Input data failed to load.")
    if dataflow_model is None: print("ERROR: Dataflow ONNX model failed to load or setup.")
    print("Cannot perform comparison due to missing components.")

elif inputs.shape[0] == 0: print("ERROR: Input data array is empty.")
else:
    num_to_run = min(NUM_SAMPLES_TO_TEST, inputs.shape[0])
    if num_to_run == 0: print("ERROR: No input samples found.")
    else:
        print(f"Comparing first {num_to_run} samples...")
        for i in range(num_to_run):
            print(f"\n--- Sample {i} ---")
            samples_compared += 1

            # 1. Prepare SLICED Float Input for PyTorch
            pytorch_input_np = None
            try:
                current_sample_full = inputs[i] # Original full sample
                #current_sample_full = np.zeros((1000,), dtype=np.float32) # Placeholder for the full sample
                current_sample_full = current_sample_full.reshape(1, 1, 1000, 1) # Reshape to (1, 1, 1000, 1) for PyTorch
                
                
                print(f"input sample {i} shape: {current_sample_full.shape}") # Expect (1, 1, 1000, 1)
                # Slice the NumPy array first (adjust axis if needed)
                # Assuming original shape is (1, C, H, W) like (1, 1, 1000, 1)
                if current_sample_full.shape == (1, 1, 1000, 1):
                    sliced_sample_np = current_sample_full#[:, :, 168:833, :] # Slice H dim -> (1, 1, 665, 1)
                    #print(sliced_sample_np)
                # Add other shape checks if necessary
                else:
                     # Try a generic slice assuming time is axis 2
                     try:
                          sliced_sample_np = current_sample_full[:, :, 168:833, :]
                          print(f"Warning: Sliced input sample {i} from {current_sample_full.shape} assuming time on axis 2")
                     except IndexError:
                          raise ValueError(f"Cannot determine slicing axis for input sample {i} shape {current_sample_full.shape}")

                # Sliced data is already NCHW float32
                pytorch_input_np = sliced_sample_np.astype(np.float32)
                pytorch_input_tensor = torch.from_numpy(pytorch_input_np).float()
                print(f"Prepared PyTorch Input (Sliced NCHW): {pytorch_input_tensor.shape}") # Expect (1, 1, 665, 1)

            except Exception as e:
                print(f"ERROR preparing PyTorch input for sample {i}: {e}")
                mismatched_samples += 1
                continue

            # 2. Run PyTorch Inference (on sliced input)
            pytorch_output_np = None
            try:
                # PyTorch model MUST NOT have internal slicing active
                with torch.no_grad():
                    pytorch_output_tensor = pytorch_model(pytorch_input_tensor)
                pytorch_output_np = pytorch_output_tensor.detach().numpy()
            except Exception as e:
                print(f"ERROR running PyTorch inference for sample {i}: {e}")
                mismatched_samples += 1
                continue

           # 3. Prepare SLICED & Quantized Input for ONNX Simulation
            onnx_sim_input = None
            input_scale_factor = 0.007948528975248337 # <-- Use the correct scale factor

            try:
                 # Start with the SLICED float NCHW data from Step 1
                 # This variable holds the single sample for the current iteration 'i'
                 onnx_input_float_nchw = pytorch_input_np[:,:,168:833,:] # Shape is (1, 1, 665, 1)

                 if INPUT_QUANTIZATION_ENABLED:
                     print(f"Quantizing ONNX input to {dataflow_input_dtype} using scale {input_scale_factor:.8f}...")

                     # Apply scaled quantization
                     if str(dataflow_input_dtype) == "INT8":
                         # Quantize: Divide by scale, round, cast to int8
                         #print(onnx_input_float_nchw[0].flatten())
                         #onnx_input_int_nchw = np.round(onnx_input_float_nchw / input_scale_factor).astype(np.int8)
                         onnx_input_int_nchw = [quantize_input(x) for x in onnx_input_float_nchw]
                         print(onnx_input_int_nchw[0].flatten())

                         # Clip to ensure values are within valid INT8 range [-128, 127]
                         #onnx_input_int_nchw = np.clip(onnx_input_int_nchw, -128, 127)

                     # Add elif here if dataflow_input_dtype could be something else (e.g., UINT8)
                     # elif str(dataflow_input_dtype) == "UINT8":
                     #    onnx_input_int_nchw = np.round(onnx_input_float_nchw / input_scale_factor).astype(np.uint8)
                     #    onnx_input_int_nchw = np.clip(onnx_input_int_nchw, 0, 255)

                     else:
                         # Handle other integer types or raise an error if unexpected
                         raise TypeError(f"Unsupported quantization type for scaled quantization: {dataflow_input_dtype}")

                 else:
                     # Fallback: Keep as float (shouldn't be used for specialized graph)
                     print("Warning: Input quantization disabled. Providing float input to ONNX.")
                     onnx_input_int_nchw = onnx_input_float_nchw.astype(np.float32)

                 # Transpose NCHW -> NHWC (Required by FINN dataflow convention)
                 # Transpose the SINGLE quantized sample
                 onnx_input_int_nhwc = np.transpose(onnx_input_int_nchw, (0, 2, 3, 1)) # Input (1,1,665,1) -> Output (1,665,1,1)
                 onnx_sim_input = np.ascontiguousarray(onnx_input_int_nhwc)

                 # This print should show dtype int8 and shape (1, 665, 1, 1)
                 print(f"Prepared ONNX Input (Sliced, Quantized, NHWC): shape {onnx_sim_input.shape}, dtype {onnx_sim_input.dtype}")

            except Exception as e:
                print(f"ERROR preparing ONNX input for sample {i}: {e}")
                mismatched_samples += 1
                continue

            # 4. Prepare Input Dict & Run ONNX Dataflow Simulation
            onnx_output_raw = None
            try:
                # Input dict uses the quantized INT8 (or other INT) NHWC tensor
                onnx_input_dict = {dataflow_input_name: onnx_sim_input}

                # Run simulation on the dataflow model (e.g., 21_...)
                onnx_output_dict = execute_onnx(dataflow_model, onnx_input_dict, return_full_exec_context=False)
                onnx_output_raw = onnx_output_dict[dataflow_output_name] # Raw integer output
                print(f"ONNX Sim Raw Output ('{dataflow_output_name}'): shape {onnx_output_raw.shape}, dtype {onnx_output_raw.dtype}")

                # Check if raw output dtype matches graph definition
                if str(dataflow_output_dtype).upper() != str(onnx_output_raw.dtype).upper():
                     # Allow compatible int types (e.g., int32 vs INT32) but warn for major diffs
                     if not (str(dataflow_output_dtype).upper().startswith("INT") and str(onnx_output_raw.dtype).upper().startswith("INT")):
                          print(f"WARNING Sample {i}: Raw ONNX output dtype {onnx_output_raw.dtype} differs significantly from expected {dataflow_output_dtype}.")

            except ImportError as e:
                 if "pyverilator" in str(e).lower(): print("\nERROR: PyVerilator missing (pip install pyverilator)\n"); sys.exit(1)
                 else: print(f"ERROR running ONNX sim (ImportError): {e}"); mismatched_samples += 1; continue
            except Exception as e:
                print(f"ERROR running ONNX dataflow simulation for sample {i}: {e}")
                # Add specific checks if needed (e.g., forrtlsim input errors)
                print(f"   Input dictionary: {{'{dataflow_input_name}': shape {onnx_input_dict[dataflow_input_name].shape}, dtype {onnx_input_dict[dataflow_input_name].dtype}}}")
                mismatched_samples += 1
                continue

             # 5. Dequantize ONNX Output using Extracted Scales and Compare
            if pytorch_output_np is None or onnx_output_raw is None:
                 print(f"ERROR Sample {i}: Missing outputs for comparison."); mismatched_samples += 1; continue

            if pytorch_output_np.size != onnx_output_raw.size:
                print(f"ERROR Sample {i}: Element count mismatch! PyTorch={pytorch_output_np.size}, ONNX={onnx_output_raw.size} (shape PyT={pytorch_output_np.shape}, ONNX={onnx_output_raw.shape})")
                mismatched_samples += 1
                continue

            try:
                # ONNX simulation output (accumulator values as float32)
                onnx_accumulator_values = onnx_output_raw # Shape (1,1,1,5)

                # Reshape accumulator to (1,5) for element-wise multiplication
                onnx_accumulator_flat = onnx_accumulator_values.flatten().reshape(1, -1) # Shape (1,5)
                print(f"ONNX Accumulator (Flat): {onnx_accumulator_flat}, dtype {onnx_accumulator_flat.dtype}")
                # --- Apply the specific scale factors ---
                onnx_output_dequantized = onnx_accumulator_flat.astype(np.float32) * scale_factors_reshaped # (1,5) * (1,5)

                # Flatten PyTorch output for comparison
                pytorch_output_flat = pytorch_output_np.flatten().astype(np.float32) # Shape (1,5)
                #print(f"PyTorch Output (Flat): {pytorch_output_np}, dtype {pytorch_output_np.dtype}")
                pytorch_output_flat_1 = pytorch_output_np.flatten().astype(np.float32) / scale_factors_reshaped# Shape (1,5)
                print(f"PyTorch Output (Flat): {pytorch_output_flat_1}, dtype {pytorch_output_flat_1.dtype}")
                
                
                onnx_output_flat = onnx_output_dequantized.flatten() # Already (1,5), just ensure flat view
                # --- Compare ---
                if not np.allclose(pytorch_output_flat, onnx_output_flat, rtol=RTOL, atol=ATOL):
                    print(f"WARNING: Numerical mismatch detected for sample {i} after dequantization!")
                    mismatched_samples += 1
                    # ... (Print PyTorch, Dequantized ONNX, Raw Accumulator, Differences) ...
                    print(f"   PyTorch Output (Flat): {pytorch_output_flat}")
                    print(f"   ONNX Output (Flat, Dequantized): {onnx_output_flat}")
                    print(f"   ONNX Sim Accumulator (as float32): {onnx_accumulator_flat}") # Print the flat accumulator
                    # ... (calculate and print max differences) ...

                else:
                   print("   Outputs match numerically after dequantization (within tolerance).") # Success message

            except Exception as e:
                print(f"ERROR during comparison/dequantization for sample {i}: {e}")
                mismatched_samples += 1

# --- Final Summary ---
print("\n--- Comparison Summary ---")
if not can_compare:
     if pytorch_model is None: print("ERROR: PyTorch model failed to load.")
     if inputs is None: print("ERROR: Input data failed to load.")
     if dataflow_model is None: print("ERROR: Dataflow ONNX model failed to load or setup.")
     print("Cannot perform comparison due to missing components.")
elif samples_compared == 0: print("ERROR: No samples were successfully processed.")
elif mismatched_samples == 0: print(f"✅ All {samples_compared} tested samples match between PyTorch ({MODEL_CHOICE}) and ONNX sim ({ONNX_MODEL_PATH}) (atol={ATOL}, rtol={RTOL}).")
else: print(f"❌ Mismatches found in {mismatched_samples}/{samples_compared} samples comparing PyTorch ({MODEL_CHOICE}) and ONNX sim ({ONNX_MODEL_PATH})!"); print(f"   Max Abs Diff: {max_abs_diff:.6f}"); print(f"   Max Rel Diff: {max_rel_diff:.6f}")
print("--------------------------")