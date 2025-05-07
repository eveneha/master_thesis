import onnxruntime as ort
import numpy as np

# Load your specific ONNX model (exported from TCN2d_inf)
onnx_model_path = "/home/eveneiha/finn/workspace/finn/onnx/19_tcn_beforePArt.onnx" # Make sure this is the correct path
#onnx_model_path = "/home/eveneiha/finn/workspace/finn/onnx/tcn_beforePArt.onnx" # Make sure this is the correct path

OUTPUT_DEQUANT_SCALE_FACTOR = 0.007948528975248337
#OUTPUT_DEQUANT_SCALE_FACTOR = 0.008209294639527798

def dequantize_input(x_int, scale=OUTPUT_DEQUANT_SCALE_FACTOR, zero_point=0):
    x_float = (x_int.astype(np.float32) - zero_point) * scale
    return x_float

def quantize_input(x_float, scale=OUTPUT_DEQUANT_SCALE_FACTOR, zero_point=0):
    x_int = np.round(x_float / scale + zero_point).astype(int)
    return x_int

# Load your identical input NumPy array
# input_data_npy = np.load("your_complex_input.npy").astype(np.float32) # Or however you load it

file_path = "/home/eveneiha/finn/workspace/ml/model/goldentb/pynq_input.npy"
#file_path = "/home/eveneiha/finn/workspace/ml/model/goldentb/pynq_input1000.npy"

x = np.load(file_path)

x = dequantize_input(x, scale=OUTPUT_DEQUANT_SCALE_FACTOR, zero_point=0)
print(x)

input_data_npy = x.reshape(1,1,665,1) # Use the array from Step 2
#input_data_npy = x.reshape(1,1,1000,1) # Use the array from Step 2


print(f"Loaded input shape: {input_data_npy.shape}, dtype: {input_data_npy.dtype}")

# import onnxruntime as ort # No longer need this
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper # Use FINN's ModelWrapper
from qonnx.core.onnx_exec import execute_onnx  # Use FINN's executor
import os # For path joining if needed

# <<< --- Point to the ONNX model you want to use as reference --- >>>
# This should ideally be the *initial* export from PyTorch (TCN2d_inf)
# before heavy FINN processing, BUT it will still contain QONNX ops.
# Let's use the one you tried: tcn_v41_inf.onnx
# Make sure the path is correct relative to where you run the script,
# or use an absolute path.
# onnx_model_path = "/home/eveneiha/finn/workspace/finn/onnx/tcn_v41_inf.onnx"
# --- OR --- Use a path relative to the script's location:


# Load the model using FINN's ModelWrapper
model = ModelWrapper(onnx_model_path)

# Get input name from the model graph
input_name = model.graph.input[0].name
# Get final output name
final_output_name = model.graph.output[0].name

# <<< --- List the EXACT intermediate tensor names recorded from Netron --- >>>
# These names MUST exist in the tcn_v41_inf.onnx graph
intermediate_tensor_names = [
    "StreamingSlice_0_out0"   # Replace with actual name from Netron
]
# Verify these tensors exist in the graph (optional but recommended)
all_tensor_names = model.get_all_tensor_names()

#for i in all_tensor_names:
   # print(f"Tensor: {i}")
for name in intermediate_tensor_names:
    if name not in all_tensor_names:
        print(f"WARNING: Intermediate tensor '{name}' not found in the ONNX graph!")
        # Consider exiting or removing the name if it's definitely wrong

# Load your identical input NumPy array
# input_data_npy = np.load("your_complex_input.npy") # Or however you load it
# --- Make sure you load the complex input here ---
# Example placeholder:
# input_data_npy = np.random.randn(1, 1, 665, 1).astype(np.float32)

# FINN's execution usually expects float32 input unless the very first node
# is a Quant node indicating integer input. Let's assume float32 for now.
# Check the model.graph.input[0] type if unsure.
input_dtype = np.float32
input_data_npy = input_data_npy.astype(input_dtype)

print(f"Running FINN execution with input shape: {input_data_npy.shape}, dtype: {input_data_npy.dtype}")

# Prepare input dictionary for FINN's executor
input_dict = {input_name: input_data_npy}

# Execute the model using FINN's executor
# return_full_exec_context=True gives ALL intermediate tensors
output_dict = execute_onnx(model, input_dict, return_full_exec_context=True)

# --- Reference values are now in the output_dict dictionary ---
# The keys are the tensor names
print("\n--- FINN execute_onnx Intermediate Results ---")


output_quant_lst = [0.00008517329843016341,0.00008318317122757435,0.00008052864723140374,0.00006355714140227064,0.00009376597154187039]

# Print final output first
i=0
if final_output_name in output_dict:
    value = output_dict[final_output_name]
    value = value / output_quant_lst[i]
    i+=1
    print(i)
    print(f"Tensor: {final_output_name} (Final Output)")
    print(f"  Shape: {value.shape}")
    print(f"  Dtype: {value.dtype}") # Note the dtype! FINN might return int
    print(f"  Sample Values (flattened): {value.flatten()[:5]}...")
    # np.save(f"finn_ref_{final_output_name.replace('/', '_')}.npy", value)
else:
     print(f"Final output {final_output_name} not found in results.")


np.set_printoptions(threshold=5000)  # or whatever default you want
np.set_printoptions(suppress=True, formatter={'float_kind': '{:,.0f}'.format})
# Print requested intermediate outputs
for name in intermediate_tensor_names:
    if name in output_dict:
        value = output_dict[name]
        print(f"Tensor: {name}")
        print(f"  Shape: {value.shape}")
        print(f"  Dtype: {value.dtype}") # Note the dtype!
        print(f"  Sample Values (flattened): {value.flatten()[:4000]}...")
        # np.save(f"finn_ref_{name.replace('/', '_')}.npy", value)
    else:
        print(f"Tensor: {name} -- NOT FOUND in execution context!")

print("-" * 30)

# You can access specific results like:
# ref_block1_relu_out = output_dict.get("tensor_name_after_block1_relu", None)