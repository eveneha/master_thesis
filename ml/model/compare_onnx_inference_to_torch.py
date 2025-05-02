
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
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat, Int8WeightPerChannelFloat, Uint8ActPerTensorFloat
from onnx import shape_inference
import logging
logging.getLogger("qonnx.core.datatype").setLevel(logging.WARNING)

# ==============================================================================
# --- Configuration - CHOOSE MODEL AND PATHS HERE ---
# ==============================================================================

# Options: "original" or "inference"
#MODEL_CHOICE = "original"
MODEL_CHOICE = "inference"
# Path to the ONNX model file you want to simulate
# This should correspond to the processed version of the chosen PyTorch model
# (e.g., the ONNX just before partitioning for the chosen architecture)
#ONNX_MODEL_PATH = "/home/eveneiha/finn/workspace/finn/onnx/tcn_before_globalavg.onnx" # <--- PATH TO ONNX TO TEST
ONNX_MODEL_PATH = "/home/eveneiha/finn/workspace/finn/onnx/07tcn_beforePArt.onnx" # <--- PATH TO ONNX TO TEST

# Path to the PyTorch checkpoint (.pth file) containing the weights
# Make sure these weights correspond to the chosen MODEL_CHOICE architecture
PYTORCH_WEIGHTS_PATH = "/home/eveneiha/finn/workspace/ml/model/tcn_model_v41_avg_pool.pth" # <--- PATH TO WEIGHTS

# Path to the input data (.pt file)
INPUT_DATA_PATH = "/home/eveneiha/finn/workspace/ml/data/preprocessed/test.pt"

NUM_SAMPLES_TO_TEST = 20
NUM_OUTPUT_CLASSES = 5

# Slicing parameters (only used if MODEL_CHOICE is "inference")
INPUT_SLICE_START = 168
INPUT_SLICE_END = 833

# Tolerances for comparison
ATOL = 1e-5
RTOL = 1e-4

# ==============================================================================
# --- PyTorch Model Definitions (Keep YOUR definitions here) ---
# ==============================================================================

# # --- Globals ---

# # Valid labels for raw input (before AAMI conversion)
# VALID_RAW_LABELS = [
#     'N', '·', 'L', 'R', 'e', 'j',
#     'A', 'a', 'J', 'S',
#     'V', 'E',
#     'F',
#     '/', 'f', 'Q', '[', ']', '!', 'x', '|'
# ]

# # AAMI Class Mapping
# AAMI_CLASSES = {
#     'N': ['N', '·', 'L', 'R', 'e', 'j'],
#     'S': ['A', 'a', 'J', 'S'],
#     'V': ['V', 'E'],
#     'F': ['F'],
#     'Q': ['/', 'f', 'Q', '[', ']', '!', 'x', '|']
# }

# # Generate AAMI_MAP from AAMI_CLASSES
# AAMI_MAP = {}
# for aami_label, raw_list in AAMI_CLASSES.items():
#     for raw_label in raw_list:
#         AAMI_MAP[raw_label] = aami_label

# AAMI_CLASS_NAMES = list(AAMI_CLASSES.keys())  # ['N', 'S', 'V', 'F', 'Q']



# # --- Split train/val/test from .pt-files ---

# save_dir = '/home/eveneiha/finn/workspace/ml/data/preprocessed'

# train_data = torch.load(os.path.join(save_dir, "train.pt"))
# train_inputs = train_data["inputs"]
# train_labels = train_data["labels"]
# train_ids = train_data["window_ids"]

# val_data = torch.load(os.path.join(save_dir, "val.pt"))
# val_inputs = val_data["inputs"]
# val_labels = val_data["labels"]
# val_ids = val_data["window_ids"]

# test_data = torch.load(os.path.join(save_dir, "test.pt"))
# test_inputs = test_data["inputs"]
# test_labels = test_data["labels"]
# test_ids = test_data["window_ids"]

# class PreprocessedECGDataset(Dataset):
#     def __init__(self, inputs, labels, win_ids):
#         self.inputs = inputs
#         self.labels = labels
#         self.win_ids = win_ids

#     def __len__(self):
#         return self.inputs.size(0)

#     def __getitem__(self, idx):
#         return self.inputs[idx], self.labels[idx], self.win_ids[idx]

# train_dataset = PreprocessedECGDataset(train_inputs, train_labels, train_ids)
# val_dataset   = PreprocessedECGDataset(val_inputs, val_labels, val_ids)
# test_dataset  = PreprocessedECGDataset(test_inputs, test_labels, test_ids)


# batch_size = 16
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataloader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# # --- Main TCN Architecture ---



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


    def forward(self, x): # Input is float
        #x = self.inp_quant(x) # Output is INT8 tensor value
        #x = x[:, :, 168:833, :] # Select the first channel
        for block in self.temporal_blocks:
            x = block(x) 
        #x = self.reduce(x)# Pass INT8 tensor value
        x = x[:, :, 84:85, :]
        x = self.fc(x) # Output likely INT32 internal or QuantTensor
        x = x.value if hasattr(x, 'value') else x
        #x = self.out_quant(x)  # Final quantization to INT8
        x = x.value if hasattr(x, 'value') else x
        x_reshaped = x.reshape(x.size(0), -1)
        return x_reshaped
    


# # --- Redefinition of TCN to TCN2d_inf for inference ---   
    
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
        #print(f"dilation: {self.conv1.dilation}")
        #print(f"dilation: {self.conv2.dilation}")

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.slice_selector_step(x) if self.slice_selector_step is not None else x

        x = self.bn2(x) #if self.last_block else self.bn2(x) # does effect model preformance 

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
                
        # Input quant layer
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
        #x = x[:, :, 168:833, :]  # 0-based index
        #qx = self.inp_quant(x)

        for block in self.temporal_blocks:
            x = block(x)
            
        x = x.value if hasattr(x, 'value') else x 
        x = self.fc(x)
        x = x.value if hasattr(x, 'value') else x
        #x = self.out_quant(x)
        x = x.value if hasattr(x, 'value') else x
        x = x.reshape(x.size(0), -1)
        
        return x
    




# --- Instantiate blocks (do this once) ---
block1_orig = TemporalBlock2d(1, 4, kernel_size=9, dilation=1,  stride=2,  dropout=0.05)
block2_orig = TemporalBlock2d(4, 8, kernel_size=9, dilation=4,  stride=1,  dropout=0.05)
block3_orig = TemporalBlock2d(8, 16, kernel_size=9, dilation=16, stride=1,  dropout=0.05)
custom_blocks_orig = [block1_orig, block2_orig, block3_orig]

block1_inf = TemporalBlock2d_inf(1, 4,  kernel_size=9,    stride=2,  dropout=0.05, use_stride = True)
block2_inf = TemporalBlock2d_inf(4, 8,  kernel_size=9,    stride=1,  dropout=0.05,  last_block=True)
block3_inf = TemporalBlock2d_inf(8, 16, kernel_size=9,    stride=1,  dropout=0.05)
custom_blocks_inf_pt = [block1_inf, block2_inf, block3_inf]

# ==============================================================================

# --- Load Data ---
print("Loading data from .pt file...")
inputs = None
try:
    data_dict = torch.load(INPUT_DATA_PATH, map_location=torch.device('cpu'), weights_only=False)
    if isinstance(data_dict, dict):
        input_key = 'inputs' # Assumes key is 'inputs'
        if input_key not in data_dict:
            possible_keys = [k for k, v in data_dict.items() if isinstance(v, (torch.Tensor, np.ndarray))]
            raise KeyError(f"Input key '{input_key}' not found. Possible keys: {possible_keys}")
        inputs_tensor = data_dict[input_key]
        if isinstance(inputs_tensor, torch.Tensor):
            inputs = inputs_tensor.detach().numpy().astype(np.float32)
        elif isinstance(inputs_tensor, np.ndarray):
            inputs = inputs_tensor.astype(np.float32)
        else:
            raise TypeError(f"Loaded input data is not Tensor/ndarray: {type(inputs_tensor)}")
        print(f"Loaded input data shape: {inputs.shape}")
    else:
         raise TypeError(f"Unexpected data type loaded: {type(data_dict)}")
except FileNotFoundError:
    print(f"ERROR: Input data file not found at {INPUT_DATA_PATH}")
except KeyError as e:
    print(f"ERROR: Could not find expected key in the loaded .pt file: {e}")
except Exception as e:
    print(f"ERROR loading or processing input data: {e}")

# --- Load Selected PyTorch Model ---
pytorch_model = None
print(f"Loading PyTorch model structure: {MODEL_CHOICE}...")
try:
    if inputs is None: raise ValueError("Input data not loaded.") # Check if data loading succeeded

    if MODEL_CHOICE == "original":
        pytorch_model = TCN2d(custom_blocks=custom_blocks_orig, num_outputs=NUM_OUTPUT_CLASSES)
        apply_input_slice = False
    elif MODEL_CHOICE == "inference":
        pytorch_model = TCN2d_inf(custom_blocks=custom_blocks_inf_pt, num_outputs=NUM_OUTPUT_CLASSES)
        apply_input_slice = False
    else:
        raise ValueError("MODEL_CHOICE must be 'original' or 'inference'")

    checkpoint = torch.load(PYTORCH_WEIGHTS_PATH, map_location=torch.device('cpu'), weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Load weights, potentially ignoring missing/unexpected keys if architectures differ slightly
    # For strict matching, remove strict=False
    missing_keys, unexpected_keys = pytorch_model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        print(f"Warning: Unexpected keys found in checkpoint for {MODEL_CHOICE} model: {unexpected_keys}")
    if missing_keys:
        print(f"Warning: Missing keys in {MODEL_CHOICE} model for checkpoint: {missing_keys}")

    pytorch_model.eval()
    print(f"PyTorch model '{MODEL_CHOICE}' loaded.")
except FileNotFoundError:
    print(f"ERROR: PyTorch weights file not found: {PYTORCH_WEIGHTS_PATH}")
    pytorch_model = None
except Exception as e:
    print(f"ERROR loading PyTorch model or weights: {e}")
    pytorch_model = None

# --- Load ONNX Model ---
print(f"Loading ONNX model: {ONNX_MODEL_PATH}...")
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

# --- Comparison Loop ---
print("\n--- Starting Comparison ---")
max_abs_diff = 0.0
max_rel_diff = 0.0
mismatched_samples = 0
samples_compared = 0
# Check if everything loaded correctly before starting
can_compare = pytorch_model is not None and inputs is not None and model_for_sim is not None

if can_compare:
    num_to_run = min(NUM_SAMPLES_TO_TEST, inputs.shape[0])
    if num_to_run == 0:
        print("ERROR: No input samples found to run comparison.")
    else:
        for i in range(num_to_run):
            print(f"\n--- Sample {i} ---")
            samples_compared += 1 # Increment here

            # 1. Prepare Input for PyTorch (Always use full original input)
            try:
                # Assuming original input needs NCHW: (1, 1, 1000, 1)
                current_sample = inputs[i]
                if current_sample.ndim == 1 and current_sample.shape[0] == 1000:
                    pytorch_input_np = current_sample.reshape(1, 1, 1000, 1)
                elif current_sample.shape == (1, 1, 1000, 1):
                    pytorch_input_np = current_sample
                else:
                    # Attempt generic reshape if possible, warn otherwise
                    try:
                         pytorch_input_np = current_sample.reshape(1, 1, 1000, 1)
                         print(f"Warning: Reshaped input sample {i} from {current_sample.shape} to {pytorch_input_np.shape}")
                    except:
                         raise ValueError(f"Cannot reshape input sample {i} (shape {current_sample.shape}) to target (1, 1, 1000, 1)")

                pytorch_input_tensor = torch.from_numpy(pytorch_input_np).float()
                # print(f"PyTorch Input Shape: {pytorch_input_tensor.shape}")
            except Exception as e:
                print(f"ERROR preparing PyTorch input for sample {i}: {e}")
                mismatched_samples += 1
                continue

            # 2. Run PyTorch Inference
            try:
                with torch.no_grad():
                    pytorch_output_tensor = pytorch_model(pytorch_input_tensor)
                pytorch_output_np = pytorch_output_tensor.detach().numpy()
                # print(f"PyTorch Output Shape: {pytorch_output_np.shape}")
            except Exception as e:
                print(f"ERROR running PyTorch inference ({MODEL_CHOICE} model) for sample {i}: {e}")
                mismatched_samples += 1
                continue

            # 3. Prepare Input for ONNX Simulation
            try:
                # Slice the input ONLY if comparing the "inference" model variant
                if MODEL_CHOICE == "inference" and apply_input_slice:
                    onnx_input_nchw = pytorch_input_np[:, :, INPUT_SLICE_START:INPUT_SLICE_END, :]
                    # print(f"ONNX Simulation Input Shape (Sliced NCHW): {onnx_input_nchw.shape}")
                else:
                    # Use the full input if comparing the original model
                    onnx_input_nchw = pytorch_input_np
                    # print(f"ONNX Simulation Input Shape (Full NCHW): {onnx_input_nchw.shape}")

                # execute_onnx needs float32 input dict
                onnx_input_dict = {input_name_onnx: onnx_input_nchw.astype(np.float32)}
            except Exception as e:
                print(f"ERROR preparing ONNX input for sample {i}: {e}")
                mismatched_samples += 1
                continue

            # 4. Run ONNX Inference (using execute_onnx)
            try:
                onnx_output_dict = execute_onnx(model_for_sim, onnx_input_dict, return_full_exec_context=False)
                onnx_output_np = onnx_output_dict[output_name_onnx]
                # print(f"ONNX Sim Output Shape: {onnx_output_np.shape}")
            except Exception as e:
                print(f"ERROR running ONNX simulation (execute_onnx) for sample {i}: {e}")
                print(f"   Input dictionary causing error: {{'{input_name_onnx}': shape {onnx_input_dict[input_name_onnx].shape}, dtype {onnx_input_dict[input_name_onnx].dtype}}}")
                mismatched_samples += 1
                continue

            # 5. Compare Outputs
            if pytorch_output_np.shape != onnx_output_np.shape:
                print(f"ERROR: Shape mismatch! PyTorch={pytorch_output_np.shape}, ONNX={onnx_output_np.shape}")
                mismatched_samples += 1
                continue

            try:
                # Compare using float32
                onnx_output_float = onnx_output_np.astype(np.float32)
                pytorch_output_float = pytorch_output_np.astype(np.float32)

                if not np.allclose(pytorch_output_float, onnx_output_float, rtol=RTOL, atol=ATOL):
                    print(f"WARNING: Numerical mismatch detected for sample {i}!")
                    mismatched_samples += 1
                    abs_diff = np.abs(pytorch_output_float - onnx_output_float)
                    rel_diff = abs_diff / (np.maximum(np.abs(pytorch_output_float), np.abs(onnx_output_float)) + 1e-9)
                    current_max_abs = abs_diff.max()
                    current_max_rel = rel_diff.max()
                    print(f"   Max Absolute Difference: {current_max_abs:.6f}")
                    print(f"   Max Relative Difference: {current_max_rel:.6f}")
                    max_abs_diff = max(max_abs_diff, current_max_abs)
                    max_rel_diff = max(max_rel_diff, current_max_rel)
                # else:
                #    print("   Outputs match numerically (within tolerance).")
            except Exception as e:
                print(f"ERROR during comparison for sample {i}: {e}")
                mismatched_samples += 1

else:
    print("Prerequisites not met (check model/data loading). Skipping comparison loop.")


# --- Final Summary ---
print("\n--- Comparison Summary ---")
if not can_compare:
     print("ERROR: Cannot perform comparison due to missing PyTorch model, input data, or ONNX model.")
elif samples_compared == 0:
     print("ERROR: No samples were successfully processed for comparison.")
elif mismatched_samples == 0:
    print(f"✅ All {samples_compared} tested samples match between PyTorch ({MODEL_CHOICE} model) and ONNX simulation (within tolerance).")
else:
    print(f"❌ Mismatches found in {mismatched_samples}/{samples_compared} samples comparing PyTorch ({MODEL_CHOICE} model) and ONNX!")
    print(f"   Overall Max Absolute Difference Found: {max_abs_diff:.6f}")
    print(f"   Overall Max Relative Difference Found: {max_rel_diff:.6f}")
print("--------------------------")