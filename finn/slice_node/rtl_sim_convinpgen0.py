import numpy as np
import os
import sys
try:
    import pyverilator
except ImportError:
    print("ERROR: PyVerilator not found. Please install it: pip install pyverilator")
    sys.exit(1)

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- Paths ---
# ** Adjust these paths based on your FINN installation **
finn_root = os.environ.get("FINN_ROOT", "/home/eveneiha/finn") # Provide a default if needed
print(f"Using FINN_ROOT: {finn_root}")

# Path to the specific Verilog file for the module under test
VERILOG_MODULE_PATH_DIR = "/home/eveneiha/finn/finn_host_build_dir/code_gen_ipgen_ConvolutionInputGenerator_rtl_0_y7viff6r"
VERILOG_FILE_NAME = os.path.join(VERILOG_MODULE_PATH_DIR, "ConvolutionInputGenerator_rtl_0_impl.sv")

# Path to the MAIN FINN RTL library directory
FINN_HDL_RTL_PATH = os.path.join(finn_root, "finn_host_build_dir/code_gen_ipgen_ConvolutionInputGenerator_rtl_0_y7viff6r") # <<< Correct path to main RTL lib

BUILD_DIR = "./pyverilator_build_ConvInputGen_0_final" # Use a new build dir name

# Check if files/dirs exist
if not os.path.isfile(VERILOG_FILE_NAME):
    print(f"ERROR: Cannot find Verilog source file: {VERILOG_FILE_NAME}.")
    sys.exit(1)
if not os.path.isdir(VERILOG_MODULE_PATH_DIR):
     print(f"ERROR: Verilog module directory not found: {VERILOG_MODULE_PATH_DIR}")
     sys.exit(1)
# Check the main FINN RTL path now
if not os.path.isdir(FINN_HDL_RTL_PATH):
    print(f"ERROR: Cannot find main FINN HDL RTL directory: {FINN_HDL_RTL_PATH}.")
    print("       Ensure FINN_ROOT is correct and the path 'src/finn/hdl/rtl' exists.")
    sys.exit(1)
else:
    print(f"Found main FINN RTL directory: {FINN_HDL_RTL_PATH}")

VERILOG_TOP_MODULE = "ConvolutionInputGenerator_rtl_0_impl"


# --- Parameters for SOFTWARE REFERENCE calculation ---
# These should reflect the *correct* behavior you expect
REF_BIT_WIDTH = 8
REF_ELEM_PER_WINDOW = 9 # Kernel size K
REF_STRIDE = 2
REF_INPUT_LEN = 665
# Correct output length calculation
REF_OUTPUT_LEN = int(((REF_INPUT_LEN + 0 - 1*(REF_ELEM_PER_WINDOW-1) - 1) / REF_STRIDE) + 1) # Should be 329
REF_LAST_WRITE_ELEM = REF_OUTPUT_LEN - 1 # Should be 328

print("--- Parameters Used for SOFTWARE REFERENCE ---")
print(f"  Input Len = {REF_INPUT_LEN}")
print(f"  Kernel K = {REF_ELEM_PER_WINDOW}")
print(f"  Stride S = {REF_STRIDE}")
print(f"  Output Len = {REF_OUTPUT_LEN}")

# --- Data Files ---
INPUT_DATA_FILE = "/home/eveneiha/finn/workspace/ml/model/goldentb/input/tb_input_data.txt"

# --- AXI Widths (Based on Verilog) ---
AXI_INPUT_WIDTH = 8 # From parameter BIT_WIDTH * SIMD * MMV_IN = 8*1*1
AXI_OUTPUT_WIDTH = 8 # From parameter BIT_WIDTH * SIMD * MMV_OUT = 8*1*1
print(f"  AXI Input Width = {AXI_INPUT_WIDTH}")
print(f"  AXI Output Width = {AXI_OUTPUT_WIDTH}")

# ==============================================================================
# --- Load Input Data ---
# ==============================================================================
print(f"\nLoading input data from: {INPUT_DATA_FILE}")
if not os.path.isfile(INPUT_DATA_FILE): sys.exit(1)
input_data_list = []
with open(INPUT_DATA_FILE, 'r') as f:
    for i, line in enumerate(f):
        if i >= REF_INPUT_LEN: break # Read exactly expected number of inputs
        line = line.strip();
        if line:
            try: input_data_list.append(int(line))
            except ValueError: print(f"Warning: Skipping non-int line: {line}")
NUM_INPUT_WORDS_LOADED = len(input_data_list)
print(f"Loaded {NUM_INPUT_WORDS_LOADED} input values.")
if NUM_INPUT_WORDS_LOADED != REF_INPUT_LEN:
     print(f"Error: Loaded {NUM_INPUT_WORDS_LOADED} values, but expected {REF_INPUT_LEN}.")
     sys.exit(1) # Exit if input count mismatch
input_data_np = np.array(input_data_list, dtype=np.int64)
packed_input_stream = input_data_list
num_input_stream_words = len(packed_input_stream)

# ==============================================================================
# --- Calculate Expected Output (Software Reference - Sliding Window) ---
# ==============================================================================
print("\nCalculating expected output stream (sliding window)...")
expected_output_stream = []
for i_out in range(REF_OUTPUT_LEN): # Use calculated correct output length
    start_idx = i_out * REF_STRIDE
    end_idx = start_idx + REF_ELEM_PER_WINDOW
    if end_idx > REF_INPUT_LEN:
        print(f"Error in reference calculation: Window extends past input data at output index {i_out}")
        break
    window = input_data_np[start_idx:end_idx]
    expected_output_stream.extend(window.tolist())
NUM_EXPECTED_OUTPUT_WORDS = len(expected_output_stream)
EXPECTED_CORRECT_COUNT_REF = REF_OUTPUT_LEN * REF_ELEM_PER_WINDOW
print(f"Generated {NUM_EXPECTED_OUTPUT_WORDS} expected output elements for reference.")
if NUM_EXPECTED_OUTPUT_WORDS != EXPECTED_CORRECT_COUNT_REF:
     print(f"Error: Reference generator produced {NUM_EXPECTED_OUTPUT_WORDS}, but expected {EXPECTED_CORRECT_COUNT_REF}.")
     sys.exit(1) # Exit if reference calculation seems wrong
expected_output_stream_np = np.array(expected_output_stream, dtype=np.int64)

print(f"Prepared {num_input_stream_words} input words. Expecting {NUM_EXPECTED_OUTPUT_WORDS} output words.")

# ==============================================================================
# --- Initialize PyVerilator Simulation ---
# ==============================================================================
# Assuming Verilog source HAS BEEN MODIFIED to have CORRECT default parameters (LAST_WRITE_ELEM=328)
# OR Verilator picks up correct synthesized values. We don't pass params here.
# Assuming Verilog source has default parameter values set correctly
print("\nInitializing PyVerilator simulation (providing both paths)...")
sim = None
try:
    sim = pyverilator.PyVerilator.build(
        VERILOG_FILE_NAME,
        verilog_path=[
            VERILOG_MODULE_PATH_DIR, # Directory with the DUT and swg_pkg
            FINN_HDL_RTL_PATH        # Directory with swg_controller, swg_buffer etc.
            ],
        build_dir=BUILD_DIR,
        top_module_name=VERILOG_TOP_MODULE
    )
    print("PyVerilator simulation initialized successfully.")
except Exception as e:
    print(f"ERROR: PyVerilator build/init failed: {e}")
    if hasattr(e, 'stdout'): print("stdout:\n", e.stdout)
    if hasattr(e, 'stderr'): print("stderr:\n", e.stderr)
    build_log_path = os.path.join(BUILD_DIR, 'obj_dir', 'verilator_compile.log')
    if os.path.exists(build_log_path):
        print("\n--- Verilator Build Log ---")
        try:
            with open(build_log_path, 'r') as f: print(f.read())
        except Exception as log_e: print(f"(Could not read log file: {log_e})")
        print("---------------------------\n")
    sys.exit(1)

# ==============================================================================
# --- Determine AXI Interface Names (Match SV Module Ports) ---
# ==============================================================================
in_tdata_pin = "in0_V_V_TDATA"
in_tvalid_pin = "in0_V_V_TVALID"
in_tready_pin = "in0_V_V_TREADY"
out_tdata_pin = "out_V_V_TDATA"
out_tvalid_pin = "out_V_V_TVALID"
out_tready_pin = "out_V_V_TREADY"
reset_pin = "ap_rst_n"
clk_pin = "ap_clk"

# ==============================================================================
# --- Run Simulation ---
# ==============================================================================
print("\nRunning RTL simulation...")
output_received = []
input_word_idx = 0
output_word_idx = 0
# Timeout based on expected + generous margin
timeout_cycles = (num_input_stream_words + NUM_EXPECTED_OUTPUT_WORDS * 2) + 5000

# Reset sequence (Active Low Reset)
print("Applying reset...")
sim.io[reset_pin] = 0
sim.clock.tick()
sim.io[reset_pin] = 1 # Deassert reset
sim.clock.tick()
print("Reset released.")

# Simulation loop
print(f"Simulating up to {timeout_cycles} cycles...")
for cycle in range(timeout_cycles):
    # Set defaults for inputs this cycle
    sim.io[in_tvalid_pin] = 0
    sim.io[in_tdata_pin] = 0 # Default value when not valid
    sim.io[out_tready_pin] = 1 # Always ready to receive output

    # Drive input if available and DUT is ready
    input_finished_sending = (input_word_idx >= num_input_stream_words)
    if not input_finished_sending:
        if sim.io[in_tready_pin] == 1: # Check TREADY *before* deciding to drive
            input_val = int(packed_input_stream[input_word_idx])
            sim.io[in_tvalid_pin] = 1
            sim.io[in_tdata_pin] = input_val
            # Increment index only if valid was high and ready was high
            input_word_idx += 1
        # else: Keep TVALID low or hold previous data if needed (current logic keeps valid low)

    # Read outputs AFTER clock tick to capture registered values
    # Get registered values from PREVIOUS cycle before clocking
    current_tvalid = sim.io[out_tvalid_pin]
    current_tdata = sim.io[out_tdata_pin]

    # Clock the simulation
    sim.clock.tick()

    # Check if output was valid and ready in the previous cycle
    if current_tvalid == 1 and sim.io[out_tready_pin] == 1: # We kept TREADY high
        output_val_raw = int(current_tdata)
        # Signed conversion for INT8 output
        output_val_signed = output_val_raw
        if REF_BIT_WIDTH < 64:
            msb_mask = 1 << (REF_BIT_WIDTH - 1)
            data_mask = (1 << REF_BIT_WIDTH) - 1
            # Check MSB for two's complement
            if (output_val_raw & msb_mask):
                 output_val_signed = (output_val_raw & data_mask) - (1 << REF_BIT_WIDTH)
        output_received.append(output_val_signed)
        output_word_idx += 1

    # Check for potential end condition (Input done, and no output for a while)
    if input_finished_sending and cycle > (num_input_stream_words + 100):
         if sim.io[out_tvalid_pin] == 0:
              stable_cycles = 0; is_stable = True
              for _ in range(50):
                   # Need to check TREADY before potentially accepting late output
                   sim.io[out_tready_pin] = 1
                   current_tvalid_late = sim.io[out_tvalid_pin]
                   current_tdata_late = sim.io[out_tdata_pin]
                   sim.clock.tick()
                   if current_tvalid_late == 1 and sim.io[out_tready_pin] == 1:
                        print(f"Warning: Output TVALID became high again (cycle {cycle+_}) after expected end.")
                        is_stable = False;
                        output_val_raw = int(current_tdata_late)
                        # signed conversion...
                        if REF_BIT_WIDTH < 64:
                             msb_mask = 1 << (REF_BIT_WIDTH - 1); data_mask = (1 << REF_BIT_WIDTH) - 1
                             if (output_val_raw & msb_mask): output_val_signed = (output_val_raw & data_mask) - (1 << REF_BIT_WIDTH)
                             else: output_val_signed = output_val_raw
                        output_received.append(output_val_signed)
                        output_word_idx += 1
                        # Don't break stable check immediately, might be multiple late values
                   stable_cycles += 1
              if is_stable:
                  print(f"Simulation potentially finished: Input done and output stable low for {stable_cycles} cycles (Cycle {cycle+stable_cycles}).")
                  break # Exit main simulation loop

    # Timeout check
    if cycle >= timeout_cycles - 1:
        print(f"ERROR: Simulation timed out after {timeout_cycles} cycles!")
        break

print("RTL simulation finished.")
print(f"  Input words sent: {input_word_idx}/{num_input_stream_words}")
print(f"  Output words received: {output_word_idx} (Expected {NUM_EXPECTED_OUTPUT_WORDS})")

# ==============================================================================
# --- Compare Results ---
# ==============================================================================
print("\n--- Comparing RTL Output vs. Expected Reference Output ---")
output_received_np = np.array(output_received, dtype=np.int64)
expected_output_stream_np = np.array(expected_output_stream, dtype=np.int64)
match = False
print(f"Expected stream length: {len(expected_output_stream_np)}")
print(f"Received stream length: {len(output_received_np)}")
compare_len = min(len(expected_output_stream_np), len(output_received_np))
if compare_len == 0 and len(expected_output_stream_np) > 0: print("ERROR: No output received!")
elif compare_len == 0 and len(expected_output_stream_np) == 0: print("Warning: No output expected or received."); match = True # Technically matches if none expected
else:
    comparison = (output_received_np[:compare_len] == expected_output_stream_np[:compare_len])
    if np.all(comparison) and len(output_received_np) == len(expected_output_stream_np):
        print(f"✅ SUCCESS: RTL Simulation output matches expected reference output! ({compare_len} elements)")
        match = True
    elif np.all(comparison) and len(output_received_np) != len(expected_output_stream_np):
         print(f"⚠️ WARNING: RTL output MATCHES expected reference for the first {compare_len} elements, but lengths differ!")
         print(f"    Expected: {len(expected_output_stream_np)}, Got: {len(output_received_np)}")
         match = False
    else:
        print("❌ FAILURE: RTL Simulation output mismatch!")
        mismatched_indices = np.where(~comparison)[0]
        print(f"  Mismatch Indices (element index): {mismatched_indices}")
        max_print = 20
        for i, idx in enumerate(mismatched_indices):
             if i >= max_print: print(f"    ... (omitting {len(mismatched_indices) - max_print} more mismatches)"); break
             print(f"    Element Idx {idx}: Expected = {expected_output_stream_np[idx]:<10d} Got = {output_received_np[idx]:<10d}")
        match = False

# Final length check only if initial comparison passed
if match and len(output_received_np) != len(expected_output_stream_np):
     print(f"❌ FAILURE: Length mismatch occurred after initial elements matched! Expected {len(expected_output_stream_np)}, Got {len(output_received_np)}")
     match = False

print("---------------------------------------------------------")
sys.exit(0 if match else 1)