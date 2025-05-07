import numpy as np
import os
from qonnx.core.datatype import DataType # For data type info
import sys
# ==============================================================================
# --- Configuration - Match VHDL Testbench & ONNX Graph ---
# ==============================================================================

# --- Input Data File from Vivado Simulation ---
INPUT_TXT_FILE = "/home/eveneiha/finn/workspace/ml/model/goldentb/input/tb_input_data.txt" # Input file used by VHDL TB

# --- Target NumPy Output File ---
OUTPUT_NPY_FILE = "pynq_input.npy"

# --- Parameters matching the HARDWARE Input Interface ---
# These define how data is packed in the TXT file and need unpacking
# Get these from the VHDL TB constants or ONNX graph input
AXI_INPUT_WIDTH    = 8      # Width of s_axis_0_tdata (e.g., 8 for INT8, PE=1)
INPUT_ELEMENT_BW   = 8      # Bitwidth of a single element (e.g., 8 for INT8)
PE_IN              = 1      # Input PE count of the first layer in hardware
INPUT_IS_SIGNED    = True   # Is the element data signed (e.g., True for INT8)?

# --- Target NUMPY Array Shape & Type (for PYNQ driver) ---
# This is the logical shape the driver expects, usually NHWC
TARGET_BATCH_SIZE  = 1      # Usually 1 sample at a time for PYNQ driver
TARGET_INPUT_HEIGHT= 665    # Sliced height (H)
TARGET_INPUT_WIDTH = 1      # Input width (W)
TARGET_INPUT_CHANNELS= 1    # Input channels (C)
# Target NumPy dtype string (e.g., "int8", "uint8")
TARGET_NUMPY_DTYPE = np.int8 # <<< --- VERIFY this matches graph input FINN DataType

TARGET_SHAPE = (TARGET_BATCH_SIZE, TARGET_INPUT_HEIGHT, TARGET_INPUT_WIDTH, TARGET_INPUT_CHANNELS) # NHWC

# ==============================================================================
# --- Helper function for Unpacking ---
# ==============================================================================
def unpack_word(packed_word, num_elements, element_bw, element_is_signed):
    """Unpacks a single wide word into constituent elements."""
    elements = []
    mask = (1 << element_bw) - 1
    for i in range(num_elements):
        val_packed = (packed_word >> (i * element_bw))
        val_unsigned = val_packed & mask
        val_signed = int(val_unsigned) # Start with unsigned interpretation
        # Convert from two's complement if signed and MSB is 1
        if element_is_signed and (val_unsigned >> (element_bw - 1)) == 1:
             val_signed = val_unsigned - (1 << element_bw)
        elements.append(val_signed)
    return elements # Returns list of Python ints

# ==============================================================================
# --- Main Script Logic ---
# ==============================================================================

print(f"Reading packed data from: {INPUT_TXT_FILE}")
if not os.path.exists(INPUT_TXT_FILE):
    print(f"ERROR: Input file not found: {INPUT_TXT_FILE}")
    sys.exit(1)

# --- Read packed data from file ---
packed_input_stream = []
try:
    with open(INPUT_TXT_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line: # Skip empty lines
                packed_input_stream.append(int(line))
except Exception as e:
    print(f"ERROR reading input file: {e}")
    sys.exit(1)

print(f"Read {len(packed_input_stream)} packed words.")

# --- Calculate expected number of elements ---
num_expected_elements = TARGET_BATCH_SIZE * TARGET_INPUT_HEIGHT * TARGET_INPUT_WIDTH * TARGET_INPUT_CHANNELS
num_expected_axi_words = num_expected_elements // PE_IN
if num_expected_elements % PE_IN != 0:
     print("Warning: Total elements not divisible by PE_IN. Check parameters.")
if len(packed_input_stream) != num_expected_axi_words:
     print(f"ERROR: Number of packed words read ({len(packed_input_stream)}) does not match expected based on shape and PE ({num_expected_axi_words}).")
     sys.exit(1)


# --- Unpack data ---
print("Unpacking data...")
unpacked_elements = []
for packed_word in packed_input_stream:
    elements = unpack_word(packed_word, PE_IN, INPUT_ELEMENT_BW, INPUT_IS_SIGNED)
    unpacked_elements.extend(elements)

print(f"Unpacked {len(unpacked_elements)} elements.")

if len(unpacked_elements) != num_expected_elements:
    print(f"ERROR: Number of unpacked elements ({len(unpacked_elements)}) does not match expected total elements ({num_expected_elements}). Check PE/BW.")
    sys.exit(1)

# --- Reshape and Cast to Target NumPy format ---
try:
    # Create NumPy array from unpacked elements
    npy_data_flat = np.array(unpacked_elements)

    # Reshape to target NHWC shape
    npy_data_reshaped = npy_data_flat.reshape(TARGET_SHAPE)

    # Cast to the final target NumPy dtype
    npy_data_final = npy_data_reshaped.astype(TARGET_NUMPY_DTYPE)

    print(f"Reshaped and cast data to shape: {npy_data_final.shape}, dtype: {npy_data_final.dtype}")

except Exception as e:
    print(f"ERROR during reshaping or casting: {e}")
    print(f"Target shape was: {TARGET_SHAPE}")
    sys.exit(1)

# --- Save the NumPy array ---
try:
    np.save(OUTPUT_NPY_FILE, npy_data_final)
    print(f"Successfully saved NumPy input array to: {OUTPUT_NPY_FILE}")
except Exception as e:
    print(f"ERROR saving NumPy file: {e}")
    sys.exit(1)

print("\nDone.")