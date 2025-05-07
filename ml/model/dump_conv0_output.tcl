# Tcl Script: dump_conv0_output.tcl (Run in Loop Method - V6)

# --- Configuration ---
set time_unit ns
set tb_top "finn_design_tb"
set dut_wrapper_inst "DUT_inst"
set block_design_inst "StreamingDataflowPartition_1_i"
set target_inst "StreamingDataflowPartition_1_ConvolutionInputGenerator_rtl_0"
set clk_signal       "/${tb_top}/ap_clk"
set rst_signal       "/${tb_top}/rst_n"
set tdata_signal     "/${tb_top}/${dut_wrapper_inst}/${block_design_inst}/StreamingDataflowPartition_1_MVAU_rtl_3_out_V_TDATA"
set tvalid_signal    "/${tb_top}/${dut_wrapper_inst}/${block_design_inst}/StreamingDataflowPartition_1_MVAU_rtl_3_out_V_TVALID"
set tready_signal    "/${tb_top}/${dut_wrapper_inst}/${block_design_inst}/StreamingDataflowPartition_1_MVAU_rtl_3_out_V_TREADY"
set output_file "conv0_output_vivado_sim.txt"
# How many clock cycles to simulate after reset
set simulation_cycles 50000 ; # Adjust as needed
# Clock period value (used with run command)
set clk_period_value 10 ; # Assuming 10ns period

# --- File Handling ---
if {[catch {open $output_file "w"} f]} {
    puts "Error: Could not open file $output_file for writing: $f"
} else {
    puts "Opened $output_file successfully."
}

# Define a procedure to check and write data (Same as V4/V5)
proc check_and_dump {} {
    global tdata_signal tvalid_signal tready_signal f rst_signal clk_signal
    set clk_val_list {}; set rst_val_list {}; set tvalid_val_list {};
    set tready_val_list {}; set tdata_val_list {};
    catch {set clk_val_list [get_value $clk_signal]}
    catch {set rst_val_list [get_value $rst_signal]}
    catch {set tvalid_val_list [get_value $tvalid_signal]}
    catch {set tready_val_list [get_value $tready_signal]}
    catch {set tdata_val_list [get_value -radix hex $tdata_signal]}
    if {[llength $clk_val_list] > 0 && [llength $rst_val_list] > 0} {
        set clk_val [lindex $clk_val_list 0]
        set rst_val [lindex $rst_val_list 0]
        # Check clock AFTER run command finishes (state should be at end of run period)
        # We check if reset is inactive. The check happens after running one cycle.
        if {$rst_val eq "0"} {
            if {[llength $tvalid_val_list] > 0 && [llength $tready_val_list] > 0} {
                set tvalid_val [lindex $tvalid_val_list 0]
                set tready_val [lindex $tready_val_list 0]
                if {$tvalid_val eq "1" && $tready_val eq "1"} {
                     # We likely want the value *before* the clock edge if checking after running.
                     # However, for simplicity let's dump the current value.
                     # More precise dumping might require more complex Tcl or VCD analysis.
                    if {[llength $tdata_val_list] > 0} {
                        set tdata_val  [lindex $tdata_val_list 0]
                        if {[catch {puts $f $tdata_val} errMsg]} { puts "Error writing: $errMsg"}
                        # flush $f ; # Optional
                    }
                }
            }
        }
    }
}


# --- Execution ---
# Run past reset phase
puts "Running past reset..."
run 100 $time_unit

# Loop and manually call the procedure
puts "Stepping through simulation using 'run' and dumping data..."
puts "Simulating $simulation_cycles clock cycles..."

for {set i 0} {$i < $simulation_cycles} {incr i} {
    # Run simulation for ONE clock cycle duration
    run $clk_period_value $time_unit

    # Check and dump the state *after* running for one cycle
    check_and_dump

    # Optional: Add a progress indicator
    if {$i % 1000 == 0 && $i > 0} {
        puts "Simulated $i cycles..."
    }
    # Optional: Check for simulation end condition if possible from VHDL->Tcl
    # if {[get_value /${tb_top}/sim_done] eq "1"} { break }
}

puts "Finished running simulation cycles."

# --- Cleanup ---
if {[info exists f]} {
    if {[catch {close $f} errMsg]} {
        puts "Error closing file $output_file: $errMsg"
    } else {
        puts "Data dumping finished. Data written to $output_file"
    }
}

# Optional: Quit simulator
# quit