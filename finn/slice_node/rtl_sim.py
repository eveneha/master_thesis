import numpy as np
import os
import sys
# Optional: Check if PyVerilator is installed early
try:
    import pyverilator
except ImportError:
    print("ERROR: PyVerilator not found. Please install it: pip install pyverilator")
    sys.exit(1)

# --- Configuration ---
# Path to the generated HLS IP directory for the StreamingSlice instance
# This directory should contain solution1/impl/ip/hdl/verilog/
HLS_IP_PATH = "/home/eveneiha/finn/finn_host_build_dir/code_gen_ipgen_StreamingSlice_hls_0b0navcn8" # <<< --- VERIFY THIS PATH
# Name of the TOP Verilog module for the StreamingSlice IP
VERILOG_TOP_MODULE = "StreamingSlice_top_Slice_0" # <<< --- VERIFY THIS NAME (usually in the .v file name)

# Slice Parameters (MUST MATCH HLS SYNTHESIS)
NUM_IN = 65
NUM_OUT = 17
START_IDX = 0
STEP = 4

# Data Type Info (MUST MATCH HLS SYNTHESIS)
# From your Verilog, the interface width is 24 bits.
# The HLS C++ used ap_int<20>. Let's assume the AXI width is 24.
AXI_INPUT_WIDTH = 24
AXI_OUTPUT_WIDTH = 24
# Determine if the data is signed based on the intended FINN datatype (e.g., INT8 -> signed)
IS_SIGNED = True # Assume signed, adjust if needed

# --- Test Data Generation ---
print("Generating test data...")
# Create a simple ramp or specific test vectors
# These should be integers representable within AXI_INPUT_WIDTH
# Example: Ramp from 0 to NUM_IN-1
input_data_int = np.arange(NUM_IN, dtype=np.int64)
# If testing signed numbers:
# input_data_int = np.arange(-(NUM_IN//2), NUM_IN - (NUM_IN//2), dtype=np.int64)

# Ensure values fit within the bit width (optional but good practice)
if IS_SIGNED:
    min_val = -(2**(AXI_INPUT_WIDTH-1))
    max_val = 2**(AXI_INPUT_WIDTH-1) - 1
else:
    min_val = 0
    max_val = 2**AXI_INPUT_WIDTH - 1
# input_data_int = np.clip(input_data_int, min_val, max_val)

# --- Calculate Expected Output (using Python/NumPy) ---
print("Calculating expected output...")
expected_data = []
output_count = 0
for idx in range(NUM_IN):
    if (idx >= START_IDX) and (((idx - START_IDX) % STEP) == 0) and (output_count < NUM_OUT):
        expected_data.append(input_data_int[idx])
        output_count += 1
expected_data = np.array(expected_data, dtype=np.int64)
print(f"Expected output ({len(expected_data)} elements): {expected_data}")

# --- Prepare for PyVerilator ---
verilog_dir = os.path.join(HLS_IP_PATH, "solution1/impl/ip/hdl/verilog")
if not os.path.isdir(verilog_dir):
    print(f"ERROR: Verilog directory not found: {verilog_dir}")
    sys.exit(1)

verilog_sources = [os.path.join(verilog_dir, f) for f in os.listdir(verilog_dir) if f.endswith(".v")]
if not verilog_sources:
    print(f"ERROR: No .v files found in {verilog_dir}")
    sys.exit(1)

# --- Define the ACTUAL build directory path ---
actual_build_dir = os.path.join(verilog_dir, "pyverilator_build_streamingslice")
print(f"Using existing PyVerilator build directory: {actual_build_dir}")

# --- Initialize PyVerilator Simulation ---
print("Initializing PyVerilator simulation (using existing build directory)...")
sim = None
try:
    # Point build() to the directory where 'make' actually ran
    sim = pyverilator.PyVerilator.build(
        verilog_sources,
        verilog_path=[verilog_dir],
        top_module_name=VERILOG_TOP_MODULE,
        build_dir=actual_build_dir # <<< --- Use the correct path ---
        # No dump args for PyVerilator 0.4.0
    )
    # Important: Check if build dir already exists, maybe skip make?
    # PyVerilator might be smart enough, but worth noting.
    print("PyVerilator simulation initialized.")
except Exception as e:
    print(f"ERROR: PyVerilator build/init failed: {e}")
    print("Ensure Verilator is installed and in PATH.")
    sys.exit(1)
    
    
   # --- Run Simulation ---
print("Running RTL simulation...")
# --- Try starting trace - might fail on 0.4.0, ignore errors for now ---
# (Keep the trace try/except blocks as they were)
try:
    sim.start_tracing() # Generic trace start
    sim.start_vcd_trace("streamingslice_trace.vcd") # Attempt VCD trace
    print("Note: Waveform trace started (may fail silently on older PyVerilator)")
except AttributeError:
    print("Note: Waveform trace commands not supported by this PyVerilator version.")
except Exception as e_trace:
     print(f"Note: Error starting waveform trace (continuing simulation): {e_trace}")


# --- Use sim.io dictionary for pin access (PyVerilator 0.4.0 style) ---

# Reset sequence
sim.io['ap_rst_n'] = 0 # Use dictionary access
sim.clock.tick()
sim.io['ap_rst_n'] = 1
sim.clock.tick()

output_received = []
input_idx = 0
output_idx = 0
timeout_cycles = NUM_IN * 10 # Increased timeout slightly, just in case

# AXI-Stream Simulation Loop using sim.io
for cycle in range(timeout_cycles):
    # Default assignments
    sim.io['in0_TVALID'] = 0
    sim.io['out_r_TREADY'] = 0 # Default: Not ready to accept output this cycle

    # Drive Input Stream
    if input_idx < NUM_IN:
        sim.io['in0_TVALID'] = 1
        sim.io['in0_TDATA'] = int(input_data_int[input_idx]) # Convert to Python int

        # Read TREADY from the DUT (output pin)
        if sim.io['in0_TREADY'] == 1:
            # print(f"Cycle {cycle}: Input sent: {input_data_int[input_idx]} (idx {input_idx})")
            input_idx += 1
        # else: print(f"Cycle {cycle}: Input waiting for TREADY (DUT value: {sim.io['in0_TREADY']})...")
    # else: print(f"Cycle {cycle}: All inputs sent.")


    # Observe Output Stream
    # Set TREADY (input pin) to indicate we are ready to receive
    sim.io['out_r_TREADY'] = 1
    # Read TVALID from the DUT (output pin)
    if sim.io['out_r_TVALID'] == 1:
        # Read TDATA (output pin) only when TVALID is high and we are ready (TREADY=1)
        output_val = sim.io['out_r_TDATA'] # .value might not be needed for dict access
        # print(f"Cycle {cycle}: Output received: {output_val} (idx {output_idx})")
        output_received.append(int(output_val)) # Convert received value to int if needed
        output_idx += 1
        # Optional: Deassert TREADY briefly if needed for testing flow control
        # sim.io['out_r_TREADY'] = 0


    # Check if done
    if output_idx >= NUM_OUT and input_idx >= NUM_IN:
         # Allow a few extra cycles for pipeline drain before breaking
         if cycle > (max(input_idx, output_idx*STEP) + 10): # Heuristic drain time
            print(f"Simulation finished condition met at cycle {cycle}.")
            break

    # Tick the clock BEFORE reading outputs for next cycle
    sim.clock.tick()


else: # Loop finished due to timeout
    print(f"ERROR: Simulation timed out after {timeout_cycles} cycles!")
    print(f"  Inputs sent: {input_idx}/{NUM_IN}")
    print(f"  Outputs received: {output_idx}/{NUM_OUT}")

# --- Try stopping trace ---
try:
    sim.stop_vcd_trace()
    sim.stop_tracing()
except AttributeError:
    pass # Ignore if stop methods don't exist
except Exception as e_trace_stop:
     print(f"Note: Error stopping waveform trace: {e_trace_stop}")

print("RTL simulation finished.")


# --- Compare Results ---
print("\n--- Comparing RTL Output vs. Expected ---")
output_received_np = np.array(output_received, dtype=np.int64)

match = False
if len(output_received_np) != len(expected_data):
    print(f"ERROR: Length mismatch! Expected {len(expected_data)}, Got {len(output_received_np)}")
else:
    comparison = (output_received_np == expected_data)
    if np.all(comparison):
        print("✅ SUCCESS: RTL Simulation output matches expected output!")
        match = True
    else:
        print("❌ FAILURE: RTL Simulation output mismatch!")
        mismatched_indices = np.where(~comparison)[0]
        print(f"  Mismatch Indices: {mismatched_indices}")
        for idx in mismatched_indices[:min(5, len(mismatched_indices))]: # Print first 5 mismatches
            print(f"    Index {idx}: Expected = {expected_data[idx]}, Got = {output_received_np[idx]}")

print("-----------------------------------------")

# Return code for scripting
sys.exit(0 if match else 1)