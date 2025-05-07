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
# Torch/Brevitas only needed if loading original data/model for reference
# import torch
# from torch.utils.data import Dataset, DataLoader, Subset
# import torch.nn as nn
# import torch.nn.functional as F
# from torchinfo import summary
# import brevitas.nn as qnn
# from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat, Int8WeightPerChannelFloat, Uint8ActPerTensorFloat

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_shapes import InferShapes # Needed?
from qonnx.custom_op.registry import getCustomOp # Needed for getCustomOp
# from qonnx.transformation.general import SortGraph # Needed?

# Define quantization function (using correct scale and zero point)
# Ensure this perfectly matches how hardware input should be quantized
def quantize_input(x_float, scale=0.007994391955435276, zero_point=0):
    # Quantize to INT8 range [-128, 127]
    # Inverse of scale
    inv_scale = 1.0 / scale
    # Scale, add zero point (if any), round, clip, cast
    x_quant = np.clip(np.round(x_float * inv_scale) + zero_point, -128, 127)
    return x_quant.astype(np.int8)

import logging
logging.getLogger("qonnx.core.datatype").setLevel(logging.WARNING)
logging.getLogger("qonnx.util.basic").setLevel(logging.WARNING)

# ==============================================================================
# --- Configuration - VERIFY THESE PATHS AND FACTORS ---
# ==============================================================================

# --- ONNX model BEFORE specialization ---
ONNX_MODEL_PATH = "/home/eveneiha/finn/workspace/finn/onnx/21_tcn_before_specialize.onnx" # <<< --- Model used for golden data generation
INPUT_DATA_PATH = "/home/eveneiha/finn/workspace/ml/data/preprocessed/test.pt" # Contains full 1000 len float signals

# --- Quantization / Dequantization ---
INPUT_QUANTIZATION_ENABLED = True # Must be true to generate correct integer inputs
INPUT_SCALE_FACTOR = 0.007948528975248337 # Verified input scale factor
INPUT_ZERO_POINT = 0 # Assuming zero point is 0 for INT8 <<< --- VERIFY if non-zero needed

NUM_SAMPLES_TO_GENERATE = 1 # How many samples to generate files for
NUM_OUTPUT_CLASSES = 5 # From model architecture

# --- Output File Locations ---
OUTPUT_DIR_GOLDENTB = "./goldentb" # Create this directory if it doesn't exist
OUTPUT_DIR_INPUT = os.path.join(OUTPUT_DIR_GOLDENTB, "input")
OUTPUT_DIR_OUTPUT = os.path.join(OUTPUT_DIR_GOLDENTB, "output")
OUTPUT_TXT_INPUT_FILE = os.path.join(OUTPUT_DIR_INPUT, "tb_input_data.txt")
OUTPUT_TXT_EXPECTED_FILE = os.path.join(OUTPUT_DIR_OUTPUT, "tb_expected_output.txt")

print(f"--- Configuration ---")
print(f"Input Data: {INPUT_DATA_PATH}")
print(f"Source ONNX Model: {ONNX_MODEL_PATH}")
print(f"Input Quantization Enabled: {INPUT_QUANTIZATION_ENABLED}")
print(f"Input Scale Factor: {INPUT_SCALE_FACTOR}")
print(f"Input Zero Point: {INPUT_ZERO_POINT}")
print(f"Num Samples to Generate: {NUM_SAMPLES_TO_GENERATE}")
print(f"Output Dir: {OUTPUT_DIR_GOLDENTB}")
print(f"--------------------")

# ==============================================================================
# --- Create Output Directories ---
# ==============================================================================
os.makedirs(OUTPUT_DIR_INPUT, exist_ok=True)
os.makedirs(OUTPUT_DIR_OUTPUT, exist_ok=True)

# ==============================================================================
# --- Load Data ---
# ==============================================================================
print("\nLoading float input data from .pt file...")
inputs = None
try:
    if not os.path.exists(INPUT_DATA_PATH): raise FileNotFoundError(f"Input data file not found at {INPUT_DATA_PATH}")
    # Assuming PyTorch is not strictly needed just for data loading if it's a simple tensor dict
    data_dict = torch.load(INPUT_DATA_PATH, map_location=torch.device('cpu'), weights_only=False) # Requires torch
    if isinstance(data_dict, dict):
        input_key = 'inputs'
        if input_key not in data_dict: raise KeyError(f"Input key '{input_key}' not found.")
        inputs_tensor = data_dict[input_key]
        if isinstance(inputs_tensor, torch.Tensor): inputs = inputs_tensor.detach().numpy().astype(np.float32)
        elif isinstance(inputs_tensor, np.ndarray): inputs = inputs_tensor.astype(np.float32)
        else: raise TypeError(f"Loaded data type error: {type(inputs_tensor)}")
        print(f"Loaded input data shape: {inputs.shape}") # Expect (N, 1, 1000, 1) or (N, 1000) etc.
    else: raise TypeError(f"Unexpected loaded data type: {type(data_dict)}")
except ImportError:
    print("WARNING: PyTorch not found. Attempting simple NumPy load if applicable.")
    try:
        inputs = np.load(INPUT_DATA_PATH) # Try if it's just a .npy saved from .pt
        print(f"Loaded input data shape (numpy): {inputs.shape}")
    except Exception as e_np:
        print(f"ERROR loading data: Cannot load without PyTorch and failed NumPy load: {e_np}"); inputs = None
except Exception as e: print(f"ERROR loading data: {e}"); inputs = None


# ==============================================================================
# --- Load ONNX Model (Before Specialization) ---
# ==============================================================================
dataflow_model = None
dataflow_input_name = None
dataflow_output_name = None
dataflow_input_dtype = None
dataflow_output_dtype = None
can_generate = inputs is not None # Check if data loaded

if can_generate:
    print(f"\nLoading ONNX model: {ONNX_MODEL_PATH}...")
    try:
        if not os.path.exists(ONNX_MODEL_PATH): raise FileNotFoundError(f"ONNX model not found at {ONNX_MODEL_PATH}")
        dataflow_model = ModelWrapper(ONNX_MODEL_PATH)
        # Clean up graph if needed (optional)
        # dataflow_model = dataflow_model.transform(InferShapes())
        # dataflow_model = dataflow_model.transform(FoldConstants())
        # dataflow_model = dataflow_model.transform(RemoveUnusedTensors())

        dataflow_input_name = dataflow_model.graph.input[0].name
        dataflow_output_name = dataflow_model.graph.output[0].name
        dataflow_input_dtype = dataflow_model.get_tensor_datatype(dataflow_input_name)
        dataflow_output_dtype = dataflow_model.get_tensor_datatype(dataflow_output_name)

        print(f"Dataflow Input Name: {dataflow_input_name}")
        print(f"Dataflow Expected Input Type (Annotation): {dataflow_input_dtype}") # Should show INT8
        print(f"Dataflow Output Name: {dataflow_output_name}")
        print(f"Dataflow Expected Output Type (Annotation): {dataflow_output_dtype}") # Should show INT accumulator type

        # Validate input type annotation
        assert str(dataflow_input_dtype).upper().startswith("INT"), \
            f"Dataflow model input '{dataflow_input_name}' is {dataflow_input_dtype}, expected an INT type annotation."
        # Validate output type annotation
        assert str(dataflow_output_dtype).upper().startswith("INT"), \
            f"Dataflow model output '{dataflow_output_name}' is {dataflow_output_dtype}, expected an INT type annotation."

        print("ONNX model loaded and validated successfully.")
    except AssertionError as e: print(f"ERROR: {e}"); dataflow_model = None; can_generate = False
    except Exception as e: print(f"ERROR loading/setting ONNX model: {e}"); dataflow_model = None; can_generate = False


# ==============================================================================
# --- Generate Golden Input/Output Files from Simulation ---
# ==============================================================================
print("\n--- Generating Golden Testbench Data ---")

if not can_generate:
     print("ERROR: Cannot generate golden data due to missing components (data or ONNX model).")
elif inputs.shape[0] == 0: print("ERROR: Input data array is empty.")
else:
    num_to_generate = min(NUM_SAMPLES_TO_GENERATE, inputs.shape[0])
    print(f"Generating data for first {num_to_generate} samples...")

    all_packed_inputs = [] # Stores packed INT8 values for tb_input_data.txt
    all_packed_outputs = [] # Stores packed INT accumulator values for tb_expected_output.txt

    # --- Get Packing Parameters ---
    in_pe = 1
    out_pe = 1
    in_bw = 8 # Based on INT8 input assumption
    in_signed = True
    out_bw = 18 # Based on INT18 accumulator assumption <<< --- VERIFY THIS
    out_signed = True # Accumulators are usually signed
    # Attempt to get parameters reliably from model if needed (more robust)
    try:
        first_node = dataflow_model.find_consumer(dataflow_input_name)
        last_node = dataflow_model.find_producer(dataflow_output_name)
        if first_node: in_bw = dataflow_model.get_tensor_datatype(first_node.input[0]).bitwidth()
        if last_node: out_bw = dataflow_model.get_tensor_datatype(last_node.output[0]).bitwidth()
        # PE extraction can be added here if needed

        in_finn_dtype = dataflow_model.get_tensor_datatype(dataflow_input_name)
        out_finn_dtype = dataflow_model.get_tensor_datatype(dataflow_output_name)
        in_bw = in_finn_dtype.bitwidth()
        in_signed = in_finn_dtype.signed()
        out_bw = out_finn_dtype.bitwidth() # Use actual output dtype bw
        out_signed = out_finn_dtype.signed()

        # Assuming PE=1 based on previous analysis
        print(f"Using Input Packing: PE={in_pe}, BW={in_bw}, Signed={in_signed}")
        print(f"Using Output Packing: PE={out_pe}, BW={out_bw}, Signed={out_signed}")
    except Exception as e:
        print(f"Warning: Could not extract all packing params from model, using defaults: {e}")
        print(f"Using Default Input Packing: PE={in_pe}, BW={in_bw}, Signed={in_signed}")
        print(f"Using Default Output Packing: PE={out_pe}, BW={out_bw}, Signed={out_signed}")


    # --- Loop through samples ---
    for i in range(num_to_generate):
        print(f"Processing sample {i}...")

        # 1. Prepare SLICED Float Input
        pytorch_input_np = None
        try:
            # Adjust reshape based on actual loaded shape if needed
            current_sample_full = inputs[i]
            if current_sample_full.ndim == 3 and current_sample_full.shape[:2] == (1, 1000): # Shape (1, 1000, 1) ?
                 current_sample_full = current_sample_full.reshape(1, 1, 1000, 1) # Reshape to NCHW
            elif current_sample_full.ndim == 2 and current_sample_full.shape[0] == 1000: # Shape (1000, 1)?
                 current_sample_full = current_sample_full.reshape(1, 1, 1000, 1) # Reshape to NCHW
            elif current_sample_full.shape == (1, 1, 1000, 1): # Already NCHW
                 pass
            else:
                 raise ValueError(f"Unexpected input sample shape: {current_sample_full.shape}")

            sliced_sample_np = current_sample_full[:, :, 168:833, :] # Slice NCHW
            pytorch_input_np = sliced_sample_np.astype(np.float32) # (1, 1, 665, 1)
        except Exception as e: print(f"  ERROR preparing sliced input: {e}"); can_generate=False; break

        # 2. Prepare SLICED, Quantized INT8 NHWC Input for ONNX sim
        onnx_sim_input = None
        try:
             onnx_input_float_nchw = pytorch_input_np
             #print(onnx_input_float_nchw[0].flatten()) # Check the float input
             onnx_input_int_nchw = quantize_input(onnx_input_float_nchw, INPUT_SCALE_FACTOR, INPUT_ZERO_POINT)
             print(onnx_input_int_nchw[0].flatten()) # Check the quantized input
             # Transpose NCHW -> NHWC
             onnx_input_int_nhwc = np.transpose(onnx_input_int_nchw, (0, 2, 3, 1))
             onnx_sim_input = np.ascontiguousarray(onnx_input_int_nhwc) # INT8 NHWC (1, 665, 1, 1)
        except Exception as e: print(f"  ERROR preparing ONNX input: {e}"); can_generate=False; break

        # 3. Run ONNX Simulation to get Golden Output (Accumulators)
        onnx_output_raw = None
        try:
            onnx_input_dict = {dataflow_input_name: onnx_sim_input}
            # Use the model loaded earlier (dataflow_model)
            onnx_output_dict = execute_onnx(dataflow_model, onnx_input_dict, return_full_exec_context=False)
            onnx_output_raw = onnx_output_dict[dataflow_output_name] # Raw accumulators (as float32)
        except Exception as e: print(f"  ERROR running ONNX sim: {e}"); can_generate=False; break

        # --- Packing Logic ---
        try:
            # Pack Input (INT8 NHWC (1,665,1,1) -> flat list of 665 INT8 values)
            input_data_flat = onnx_sim_input.flatten() # Flatten the INT8 NHWC array
            num_in_elements = len(input_data_flat)
            num_in_axi_words = num_in_elements // in_pe

            sample_packed_inputs = []
            current_element_idx_in = 0
            for _ in range(num_in_axi_words):
                 word = 0
                 for pe_idx in range(in_pe):
                     if current_element_idx_in < num_in_elements:
                          # Get the ACTUAL signed INT8 value
                          val = int(input_data_flat[current_element_idx_in])
                          # --- Write the signed integer value directly ---
                          if in_pe == 1:
                               word = val
                          else: # Handle PE > 1 packing if necessary
                               mask = (1 << in_bw) - 1
                               temp_val = val
                               if in_signed and val < 0: temp_val = (1 << in_bw) + val
                               word |= (temp_val & mask) << (pe_idx * in_bw)
                          current_element_idx_in += 1
                     else: break # Should not happen
                 sample_packed_inputs.append(word)
            # Extend the main list
            all_packed_inputs.extend(sample_packed_inputs)


            # Pack Output (Accumulators, e.g., INT18/32 -> packed int for file)
            onnx_output_flat = onnx_output_raw.flatten() # Flatten (1,1,1,5) -> (5,)
            num_out_elements = len(onnx_output_flat)
            num_out_axi_words = num_out_elements // out_pe

            sample_packed_outputs = []
            current_element_idx_out = 0
            for _ in range(num_out_axi_words):
                 word = 0
                 for pe_idx in range(out_pe):
                     if current_element_idx_out < num_out_elements:
                         # Use the raw accumulator value (may be float but represents int)
                         val = int(np.round(onnx_output_flat[current_element_idx_out]))
                         # Convert to two's complement bit pattern using ACCUMULATOR bitwidth
                         mask = (1 << out_bw) - 1
                         val_bits = val & mask # Get lower bits
                         if out_signed and val < 0:
                              # Check if value fits before trying 2's comp
                              min_acc_val = -(2**(out_bw-1))
                              if val < min_acc_val: print(f"Warning: Accumulator value {val} below range for {out_finn_dtype.name}")
                              val_bits = (1 << out_bw) + val # Calculate 2's comp pattern
                         else:
                              max_acc_val = (2**(out_bw-1)) - 1 if out_signed else (2**out_bw) - 1
                              if val > max_acc_val: print(f"Warning: Accumulator value {val} above range for {out_finn_dtype.name}")

                         # Pack based on AXI output width and PE
                         axi_mask = (1 << out_bw) - 1 # Mask for individual element width
                         word |= (val_bits & axi_mask) << (pe_idx * out_bw) # Pack using element BW
                         current_element_idx_out += 1
                     else: break # Should not happen
                 sample_packed_outputs.append(word) # Append packed word representing accumulator(s)
            # Extend the main list
            all_packed_outputs.extend(sample_packed_outputs)


        except Exception as e:
            print(f"  ERROR during data packing for sample {i}: {e}")
            can_generate = False; break

    # --- Write Files After Loop ---
    if can_generate and len(all_packed_inputs) > 0:
        # Write packed data to files
        try:
            print(f"\nWriting data for {num_to_generate} samples...")
            with open(OUTPUT_TXT_INPUT_FILE, "w") as f:
                for val_int in all_packed_inputs: # Should contain N*665 values
                    f.write(f"{int(val_int)}\n") # Write the signed INT8 value
            print(f"Generated {OUTPUT_TXT_INPUT_FILE} ({len(all_packed_inputs)} words)")

            with open(OUTPUT_TXT_EXPECTED_FILE, "w") as f:
                for word in all_packed_outputs: # Should contain N*5 values
                    f.write(f"{int(word)}\n") # Write packed accumulator value
            print(f"Generated {OUTPUT_TXT_EXPECTED_FILE} ({len(all_packed_outputs)} words)")
            print("--- Golden Testbench Data Generation Complete ---")

        except Exception as e:
            print(f"ERROR writing data files: {e}")

# --- Final Summary ---
print("\n--- Script Finished ---")
if not can_generate:
     print("ERROR: Golden data generation could not complete due to errors.")
print("---------------------")