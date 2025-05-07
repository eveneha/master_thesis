import numpy as np
import os
import sys
try:
    import pyverilator
except ImportError:
    print("ERROR: PyVerilator not found. Please install it: pip install pyverilator")
    sys.exit(1)

# Re-import necessary QONNX/FINN components if needed for DataType later
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.core.datatype import DataType

# ==============================================================================
# --- Configuration - Using Provided Values ---
# ==============================================================================

# --- Paths ---
# Path to the ONNX model containing Thresholding_4 with 8 channels
ONNX_MODEL_WITH_THRESHOLDS = "/home/eveneiha/finn/workspace/finn/onnx/16_after_numInputVectors_change.onnx"
ONNX_NODE_NAME = "Thresholding_4"

# --- Location of FINN Library ---
# Try to find the FINN source directory
finn_root = os.environ.get("FINN_ROOT")
if finn_root is None:
     script_dir = os.path.dirname(os.path.realpath(__file__))
     potential_root = os.path.abspath(os.path.join(script_dir, '../../..'))
     if os.path.exists(os.path.join(potential_root, "src/finn")):
          finn_root = potential_root
          print(f"Guessed FINN_ROOT: {finn_root}")
     else:
          print("ERROR: FINN_ROOT environment variable not set or cannot be guessed.")
          sys.exit(1)
RTL_SRC_PATH = os.path.join(finn_root, "src/finn/hdl/rtl")

# --- Verilog Details ---
VERILOG_FILE_NAME = "thresholding.sv"
VERILOG_TOP_MODULE = "thresholding" # Assumed standard name

# Check if RTL source file exists
verilog_source_file = os.path.join(RTL_SRC_PATH, VERILOG_FILE_NAME)
if not os.path.isfile(verilog_source_file):
     print(f"ERROR: Verilog source file not found: {verilog_source_file}")
     sys.exit(1)

# --- Node Parameters (Using values you provided/confirmed) ---
INPUT_BIT_WIDTH = 32  # You provided this value
OUTPUT_BIT_WIDTH = 8   # You provided this value
NUM_CHANNELS_CONFIGURED = 8 # As set in the ONNX node attribute
PE = 1                  # You provided this value
NUM_THRESH_STEPS = 255 # Calculated as (1 << 8) - 1
NUM_VECTORS = 17       # Derived from [1, 17, 1] attribute
INPUT_IS_SIGNED = True  # You provided this value
OUTPUT_IS_SIGNED = False # You provided this value (UINT8 usually)
ACT_VAL = 0             # Assuming default activation value is 0 <<< --- VERIFY if needed

# --- Threshold Data Type (Need this for Verilator param) ---
# Load model just to get weightDataType attribute for Thresholding_4
print(f"Loading ONNX model for threshold dtype: {ONNX_MODEL_WITH_THRESHOLDS}")
if not os.path.exists(ONNX_MODEL_WITH_THRESHOLDS):
     print(f"ERROR: ONNX model not found: {ONNX_MODEL_WITH_THRESHOLDS}"); sys.exit(1)
model_temp = ModelWrapper(ONNX_MODEL_WITH_THRESHOLDS)
thresh_node_temp = model_temp.get_node_from_name(ONNX_NODE_NAME)
if thresh_node_temp is None: print(f"ERROR: Node {ONNX_NODE_NAME} not found"); sys.exit(1)
thresh_inst_temp = getCustomOp(thresh_node_temp)
weight_finn_dtype = thresh_inst_temp.get_weight_datatype()
THRESH_BIT_WIDTH = weight_finn_dtype.bitwidth()
THRESH_IS_SIGNED = weight_finn_dtype.signed()
OUTPUT_DT_NAME = thresh_inst_temp.get_output_datatype().name # Get precise output DT name
del model_temp, thresh_node_temp, thresh_inst_temp # Clean up temporary model load

print(f"Parameters Used:")
print(f"  Input BW={INPUT_BIT_WIDTH}, Signed={INPUT_IS_SIGNED}")
print(f"  Output BW={OUTPUT_BIT_WIDTH}, Signed={OUTPUT_IS_SIGNED}, DT={OUTPUT_DT_NAME}")
print(f"  Threshold BW={THRESH_BIT_WIDTH}, Signed={THRESH_IS_SIGNED}, Steps={NUM_THRESH_STEPS}")
print(f"  NumChannels={NUM_CHANNELS_CONFIGURED}, PE={PE}, NumVectors={NUM_VECTORS}, ActVal={ACT_VAL}")

# AXI Interface Widths
AXI_INPUT_WIDTH = INPUT_BIT_WIDTH * PE
AXI_OUTPUT_WIDTH = OUTPUT_BIT_WIDTH * PE
print(f"  AXI Input Width={AXI_INPUT_WIDTH}, AXI Output Width={AXI_OUTPUT_WIDTH}")

# ==============================================================================
# --- Load Threshold Data ---
# ==============================================================================
print(f"\nLoading thresholds from: {ONNX_MODEL_WITH_THRESHOLDS}")
model = ModelWrapper(ONNX_MODEL_WITH_THRESHOLDS) # Load model again for thresholds
threshold_initializer_name = model.get_node_from_name(ONNX_NODE_NAME).input[1] # Get threshold tensor name from node
threshold_tensor = model.get_initializer(threshold_initializer_name)
if threshold_tensor is None:
     print(f"ERROR: Could not find threshold initializer: {threshold_initializer_name}")
     sys.exit(1)

ACTUAL_THRESH_CHANNELS = threshold_tensor.shape[0]
print(f"Threshold tensor shape: {threshold_tensor.shape}")
if ACTUAL_THRESH_CHANNELS != NUM_CHANNELS_CONFIGURED:
     print(f"ERROR: Threshold data channels ({ACTUAL_THRESH_CHANNELS}) mismatch configured channels ({NUM_CHANNELS_CONFIGURED})!")
     sys.exit(1) # Exit because parameters are inconsistent

# Thresholds should match THRESH_BIT_WIDTH and THRESH_IS_SIGNED
thresholds = threshold_tensor.astype(np.int64).reshape(ACTUAL_THRESH_CHANNELS, -1)
print(f"Loaded thresholds for {ACTUAL_THRESH_CHANNELS} channels.")

# ==============================================================================
# --- Test Data Generation ---
# ==============================================================================
print("\nGenerating test input data...")
total_input_elements = NUM_VECTORS * NUM_CHANNELS_CONFIGURED
input_data_flat = np.zeros(total_input_elements, dtype=np.int64)

if INPUT_IS_SIGNED:
    min_in_val = -(2**(INPUT_BIT_WIDTH-1))
    max_in_val = 2**(INPUT_BIT_WIDTH-1) - 1
else:
    min_in_val = 0
    max_in_val = 2**INPUT_BIT_WIDTH - 1

input_data_flat = np.random.randint(min_in_val, max_in_val + 1, size=total_input_elements)
input_data_vectors = input_data_flat.reshape(NUM_VECTORS, NUM_CHANNELS_CONFIGURED)
print(f"Generated input data shape: {input_data_vectors.shape}")

# ==============================================================================
# --- Calculate Expected Output (Software Reference) ---
# ==============================================================================
print("\nCalculating expected output...")

def software_threshold(val, thresholds_for_channel, act_val, out_dt_name):
    output = act_val
    for threshold in thresholds_for_channel:
        if val > threshold:
            output += 1
        else:
            break
    out_dt = DataType[out_dt_name]
    out_min, out_max = out_dt.min(), out_dt.max()
    return np.clip(output, out_min, out_max)

expected_data_flat = np.zeros_like(input_data_flat)
for i in range(total_input_elements):
    channel_idx = i % NUM_CHANNELS_CONFIGURED
    val = input_data_flat[i]
    thresholds_for_channel = thresholds[channel_idx] # Use correct channel index
    expected_data_flat[i] = software_threshold(val, thresholds_for_channel, ACT_VAL, OUTPUT_DT_NAME)

expected_data_vectors = expected_data_flat.reshape(NUM_VECTORS, NUM_CHANNELS_CONFIGURED)
print(f"Expected output data shape: {expected_data_vectors.shape}")

# ==============================================================================
# --- Prepare Data for AXI-Stream Simulation ---
# ==============================================================================
print("\nPacking data for simulation...")
if NUM_CHANNELS_CONFIGURED % PE != 0:
    print(f"ERROR: NumChannels ({NUM_CHANNELS_CONFIGURED}) not divisible by PE ({PE})")
    sys.exit(1)
simd = NUM_CHANNELS_CONFIGURED // PE
input_data_folded = input_data_vectors.reshape(NUM_VECTORS, simd, PE)

packed_input_stream = []
for vec in range(NUM_VECTORS):
    for fold in range(simd):
        word = 0
        for pe_idx in range(PE):
            val = int(input_data_folded[vec, fold, pe_idx])
            mask = (1 << INPUT_BIT_WIDTH) - 1
            if INPUT_IS_SIGNED and val < 0: val = (1 << INPUT_BIT_WIDTH) + val
            word |= (val & mask) << (pe_idx * INPUT_BIT_WIDTH)
        packed_input_stream.append(word)
num_input_stream_words = len(packed_input_stream)

expected_data_folded = expected_data_vectors.reshape(NUM_VECTORS, simd, PE)
packed_expected_stream = []
for vec in range(NUM_VECTORS):
     for fold in range(simd):
         word = 0
         for pe_idx in range(PE):
             val = int(expected_data_folded[vec, fold, pe_idx])
             mask = (1 << OUTPUT_BIT_WIDTH) - 1
             if OUTPUT_IS_SIGNED and val < 0: val = (1 << OUTPUT_BIT_WIDTH) + val
             word |= (val & mask) << (pe_idx * OUTPUT_BIT_WIDTH)
         packed_expected_stream.append(word)
num_output_stream_words = len(packed_expected_stream)
print(f"Prepared {num_input_stream_words} input words ({AXI_INPUT_WIDTH}-bit) and {num_output_stream_words} expected output words ({AXI_OUTPUT_WIDTH}-bit).")

# ==============================================================================
# --- Prepare for PyVerilator ---
# ==============================================================================
verilog_sources = [verilog_source_file]
build_dir_name = f"pyverilator_build_{VERILOG_TOP_MODULE}_I{INPUT_BIT_WIDTH}O{OUTPUT_BIT_WIDTH}P{PE}C{NUM_CHANNELS_CONFIGURED}"
actual_build_dir = os.path.join("./", build_dir_name)
print(f"Using PyVerilator build directory: {actual_build_dir}")

# ==============================================================================
# --- Initialize PyVerilator Simulation ---
# ==============================================================================
print("\nInitializing PyVerilator simulation...")
sim = None
try:
    # Define Verilator parameters (-G flags) - **VERIFY THESE against thresholding.v**
    verilator_args = [
        f"-GIDW={INPUT_BIT_WIDTH}",      # Input Data Width per element
        f"-GODW={OUTPUT_BIT_WIDTH}",     # Output Data Width per element
        f"-GPE={PE}",                  # Parallelization Element count
        f"-GTHRESH_WIDTH={THRESH_BIT_WIDTH}", # Threshold value bit width
        f"-GTHRESH_SIGNED={'1' if THRESH_IS_SIGNED else '0'}", # Are thresholds signed?
        f"-GN_THRESH={NUM_THRESH_STEPS}", # Number of threshold levels (steps)
        f"-GACTVAL={ACT_VAL}",         # Activation value offset
        f"-GOUT_SIGNED={'1' if OUTPUT_IS_SIGNED else '0'}", # Is output signed?
        f"-GNCH={NUM_CHANNELS_CONFIGURED}", # Number of Channels
        f"-GSIMD={simd}"                # SIMD width (Channels/PE)
    ]
    print(f"Verilator Args: {verilator_args}")

    sim = pyverilator.PyVerilator.build(
        verilog_sources,
        verilog_path=[RTL_SRC_PATH],
        top_module_name=VERILOG_TOP_MODULE,
        build_dir=actual_build_dir,
        verilator_args=verilator_args,
        # dump_vcd=True # Optional: enable VCD trace
    )
    print("PyVerilator simulation initialized.")
except Exception as e:
    print(f"ERROR: PyVerilator build/init failed: {e}")
    sys.exit(1)

# ==============================================================================
# --- Determine AXI Interface Names (Standard FINN RTL Naming) ---
# ==============================================================================
axi_in_prefix = "in0_V"
axi_out_prefix = "out_V"
in_tdata_pin = f"{axi_in_prefix}_TDATA"
in_tvalid_pin = f"{axi_in_prefix}_TVALID"
in_tready_pin = f"{axi_in_prefix}_TREADY"
out_tdata_pin = f"{axi_out_prefix}_TDATA"
out_tvalid_pin = f"{axi_out_prefix}_TVALID"
out_tready_pin = f"{axi_out_prefix}_TREADY"
reset_pin = "ap_rst_n"
clk_pin = "ap_clk"

# ==============================================================================
# --- Run Simulation ---
# ==============================================================================
print("\nRunning RTL simulation...")
# (Optional trace setup)
# try: sim.start_tracing(); sim.start_vcd_trace("thresholding_rtl_trace.vcd")
# except Exception: print("Note: Waveform trace not started/supported.")

output_received_packed = []
input_word_idx = 0
output_word_idx = 0
# More generous timeout for potentially complex RTL
timeout_cycles = (num_input_stream_words + num_output_stream_words) * 5 + 500

# Reset sequence
sim.io[reset_pin] = 0
sim.clock.tick()
sim.io[reset_pin] = 1
sim.clock.tick()

# Simulation loop
for cycle in range(timeout_cycles):
    sim.io[in_tvalid_pin] = 0
    sim.io[in_tdata_pin] = 0
    sim.io[out_tready_pin] = 1 # Always ready to receive output

    # Drive input if available
    if input_word_idx < num_input_stream_words:
        sim.io[in_tvalid_pin] = 1
        sim.io[in_tdata_pin] = int(packed_input_stream[input_word_idx])
        # Check if DUT accepted input
        if sim.io[in_tready_pin] == 1:
            input_word_idx += 1

    # Check for valid output
    if sim.io[out_tvalid_pin] == 1 and sim.io[out_tready_pin] == 1:
        output_word = int(sim.io[out_tdata_pin])
        output_received_packed.append(output_word)
        output_word_idx += 1

    # Check for completion
    if input_word_idx >= num_input_stream_words and output_word_idx >= num_output_stream_words:
         # Check if output stream is potentially empty/stable
         if sim.io[out_tvalid_pin] == 0:
              # Allow a few more cycles to ensure no late output
              stable_cycles = 0
              for _ in range(10):
                   sim.clock.tick()
                   if sim.io[out_tvalid_pin] == 1: break # False alarm
                   stable_cycles += 1
              if stable_cycles == 10:
                   print(f"Simulation finished condition met at cycle {cycle+10}.")
                   break

    sim.clock.tick()
else: # Timeout
    print(f"ERROR: Simulation timed out after {timeout_cycles} cycles!")
    print(f"  Input words sent: {input_word_idx}/{num_input_stream_words}")
    print(f"  Output words received: {output_word_idx}/{num_output_stream_words}")

# (Optional trace stop)
print("RTL simulation finished.")

# ==============================================================================
# --- Compare Packed Results ---
# ==============================================================================
print("\n--- Comparing Packed RTL Output vs. Expected Packed Output ---")
output_received_packed_np = np.array(output_received_packed, dtype=np.int64)
expected_packed_stream_np = np.array(packed_expected_stream, dtype=np.int64)

match = False
if len(output_received_packed_np) != len(expected_packed_stream_np):
    print(f"ERROR: Packed length mismatch! Expected {len(expected_packed_stream_np)}, Got {len(output_received_packed_np)}")
    # Optionally print partial results
    print(f"Received: {output_received_packed_np}")
    print(f"Expected: {expected_packed_stream_np}")
else:
    comparison = (output_received_packed_np == expected_packed_stream_np)
    if np.all(comparison):
        print("✅ SUCCESS: Packed RTL Simulation output matches expected packed output!")
        match = True
    else:
        print("❌ FAILURE: Packed RTL Simulation output mismatch!")
        mismatched_indices = np.where(~comparison)[0]
        print(f"  Mismatch Indices (packed words): {mismatched_indices}")
        for idx in mismatched_indices[:min(10, len(mismatched_indices))]: # Print more mismatches
             print(f"    Word Idx {idx}: Expected = {expected_packed_stream_np[idx]:<15x} Got = {output_received_packed_np[idx]:<15x}")

# ==============================================================================
# --- Optional: Unpack and Compare Element-wise ---
# ==============================================================================
# Only unpack if packed comparison succeeded, otherwise element-wise will also fail
# if match: # <<<----- Change this if you want to see unpacked diffs even if packed fails
if len(output_received_packed_np) == len(expected_packed_stream_np): # Check lengths are same for unpacking
    print("\n--- Unpacking and comparing element-wise ---")
    output_unpacked_flat = []
    for word in output_received_packed_np:
         for pe_idx in range(PE):
              val_packed = (word >> (pe_idx * OUTPUT_BIT_WIDTH))
              mask = (1 << OUTPUT_BIT_WIDTH) - 1
              val_unsigned = val_packed & mask
              val_signed = val_unsigned
              if OUTPUT_IS_SIGNED and (val_unsigned >> (OUTPUT_BIT_WIDTH - 1)) == 1:
                   val_signed = val_unsigned - (1 << OUTPUT_BIT_WIDTH)
              output_unpacked_flat.append(val_signed)
    output_unpacked_np = np.array(output_unpacked_flat, dtype=np.int64)

    comparison_unpacked = (output_unpacked_np == expected_data_flat)
    if np.all(comparison_unpacked):
         print("✅ SUCCESS: Unpacked element-wise comparison also matches!")
         # Keep match = True if it was already True
    else:
         print("❌ FAILURE: Unpacked element-wise comparison shows mismatch!")
         mismatched_indices_unpacked = np.where(~comparison_unpacked)[0]
         print(f"  Mismatch Indices (elements): {mismatched_indices_unpacked}")
         for idx in mismatched_indices_unpacked[:min(20, len(mismatched_indices_unpacked))]: # Print more mismatches
              print(f"    Element Idx {idx}: Expected = {expected_data_flat[idx]}, Got = {output_unpacked_np[idx]}")
         match = False # Set match to False if unpacked fails

print("---------------------------------------------------------")
sys.exit(0 if match else 1)