# --- START OF FILE streamingslice_hls.py ---
import os
import shutil # Import shutil for directory removal
from finn.custom_op.fpgadataflow.streaming_slice import StreamingSlice
from finn.util.basic import make_build_dir
import warnings # Import warnings

class StreamingSlice_hls(StreamingSlice):
    # ... (other methods like __init__ if they exist) ...

    def ipgen_singlenode_code(self, build_dir):
        fpgapart = self.get_nodeattr("fpgapart")
        clk_ns = self.get_nodeattr("clk_ns")
        node = self.onnx_node
        node_name = node.name

        # --- Get Node Attributes ---
        start_idx = self.get_nodeattr("start_idx")
        slice_length = self.get_nodeattr("slice_length") # This is NumH_Out for the HLS template
        step = self.get_nodeattr("step")               # This is StepH for the HLS template
        axis = self.get_nodeattr("axis")
        input_shape = self.get_nodeattr("input_shape")
        # input_len should be the size of the dimension being sliced (NumH_In)
        input_len_on_axis = input_shape[axis]          # This is NumH_In for the HLS template
        # --- Fetch the new NumChannels attribute ---
        num_channels = self.get_nodeattr("num_channels")
        # Ensure num_channels was inferred correctly (add check)
        if num_channels is None or num_channels <= 0:
            # Try to infer it again here if needed, or raise error
            # This relies on infer_node_shape having run successfully before
            raise ValueError(f"num_channels attribute not set or invalid for node {node_name}")
        # --- End Fetch ---
        module_name = self.get_nodeattr("module_name") # HLS top function name

        inp_finn_dtype = self.get_input_datatype(0)
        out_finn_dtype = self.get_output_datatype(0)
        assert inp_finn_dtype == out_finn_dtype, f"StreamingSlice {node_name}: Input/Output DataTypes must match."
        print(f"StreamingSlice {node_name}: Input/Output FINN DataType: {inp_finn_dtype}")
        dtype_hls_str = inp_finn_dtype.get_hls_datatype_str()
        print(f"StreamingSlice {node_name}: HLS DataType: {dtype_hls_str}")
        print(f"StreamingSlice {node_name}: NumChannels passed to HLS: {num_channels}")

        # --- Generate C++ Code ---
        # Assuming streaming_slice.hpp now has the corrected template
        # named StreamingSlice_Corrected requiring NumChannels parameter
        hpp_file_name = "streaming_slice.hpp" # Use the name of your corrected header
        # Make sure to pass the correct template parameters IN ORDER
        cpp_code = f"""#include "{hpp_file_name}"
    #include "ap_int.h"
    #include "hls_stream.h"

    extern "C" {{
    void {module_name}(
        hls::stream<{dtype_hls_str}> &in0,
        hls::stream<{dtype_hls_str}> &out
        ) {{
    #pragma HLS INTERFACE axis register_mode=both port=in0 name=in0
    #pragma HLS INTERFACE axis register_mode=both port=out name=out_r
    #pragma HLS INTERFACE ap_ctrl_none port=return

        // Instantiate the *corrected* HLS template with all required parameters
        StreamingSlice<
            {dtype_hls_str},    // T (DataType)
            {input_len_on_axis},// NumH_In (Size of sliced dimension)
            {num_channels},     // NumChannels (Size of inner interleaved dimension)
            {slice_length},     // NumH_Out (Slice length along axis)
            {start_idx},        // StartH_Idx
            {step}              // StepH
        >(in0, out);
    }}
    }}"""
        cpp_path = os.path.join(build_dir, f"{module_name}.cpp")
        with open(cpp_path, "w") as f:
            f.write(cpp_code)
            print(f"Generated C++ wrapper: {cpp_path}")

        # Copy the *corrected* header file
        hpp_src = os.path.join(os.path.dirname(__file__), hpp_file_name)
        hpp_dst = os.path.join(build_dir, hpp_file_name)
        if not os.path.exists(hpp_dst) or not os.path.samefile(hpp_src, hpp_dst):
            print(f"Copying {hpp_src} to {hpp_dst}")
            shutil.copy(hpp_src, hpp_dst) # Use shutil.copy for clarity

        # --- Define Paths ---
        final_exported_ip_dir = os.path.join(build_dir, "ip") # Unzipped IP location

        # --- Vitis HLS Tcl Script ---
        # Add the corrected HPP file to add_files
        tcl_script = f"""open_project {module_name}_prj
    set_top {module_name}
    add_files {module_name}.cpp
    add_files {hpp_file_name} -cflags "-I./"
    open_solution "solution1" -flow_target vivado
    set_part {fpgapart}
    create_clock -period {clk_ns} -name default
    csynth_design
    # Export to build_dir, Vivado expects export.zip for IP catalog
    export_design -format ip_catalog -output "{build_dir}/export.zip" -rtl verilog
    exit
    """
        tcl_path = os.path.join(build_dir, "run_hls.tcl")
        with open(tcl_path, "w") as f:
            f.write(tcl_script)
            print(f"Generated HLS Tcl script: {tcl_path}")

        # --- Run Vitis HLS ---
        hls_cmd = f"cd {build_dir} && vitis_hls -f run_hls.tcl"
        print(f"Running HLS command: {hls_cmd}")
        ret = os.system(hls_cmd)

        # --- Post-HLS Checks ---
        print(f"HLS exit code: {ret}")
        if ret != 0:
            raise Exception(f"❌ Vitis HLS failed for {node_name} with exit code {ret}")

        # --- Find and Unzip the IP Archive ---
        expected_zip_path = os.path.join(build_dir, "export.zip")

        if not os.path.isfile(expected_zip_path):
             print(f"ERROR: Could not find {expected_zip_path} after HLS run.")
             print(f"Contents of build directory ({build_dir}):")
             if os.path.exists(build_dir): print(os.listdir(build_dir))
             raise FileNotFoundError("HLS export.zip archive not found in build directory.")

        print(f"Found IP archive: {expected_zip_path}")
        print(f"Unzipping to: {final_exported_ip_dir}")

        if os.path.exists(final_exported_ip_dir):
             print(f"Removing existing directory: {final_exported_ip_dir}")
             shutil.rmtree(final_exported_ip_dir)
        # Creating the directory is often handled by unzip, but be sure
        try:
            os.makedirs(final_exported_ip_dir, exist_ok=True) # Create if doesn't exist
        except OSError as e:
             raise OSError(f"Failed to create directory {final_exported_ip_dir}: {e}")

        unzip_cmd = f"unzip -oq {expected_zip_path} -d {final_exported_ip_dir}" # Added -q for quiet
        print(f"Running unzip: {unzip_cmd}")
        ret_unzip = os.system(unzip_cmd)
        if ret_unzip != 0:
             raise Exception(f"❌ Failed to unzip HLS IP archive: {expected_zip_path}")

        component_xml = os.path.join(final_exported_ip_dir, "component.xml")
        if not os.path.isfile(component_xml):
             print(f"ERROR: component.xml not found in {final_exported_ip_dir} after unzipping.")
             print(f"Contents of IP directory:")
             if os.path.exists(final_exported_ip_dir): print(os.listdir(final_exported_ip_dir))
             raise FileNotFoundError("component.xml not found after unzip.")

        print(f"✅ Generated and unzipped StreamingSlice HLS IP in {final_exported_ip_dir}")

    def code_generation_ipgen(self, model, fpgapart, clk_ns):
        """Orchestrates HLS IP generation for the StreamingSlice node."""
        node = self.onnx_node
        # Ensure module_name is set before generating code
        if not self.get_nodeattr("module_name"):
             self.set_nodeattr("module_name", self.hls_sname(node.name)) # Pass node_name

        build_dir = make_build_dir(prefix="code_gen_ipgen_" + node.name)

        self.set_nodeattr("code_gen_dir_ipgen", build_dir)
        self.set_nodeattr("ipgen_path", build_dir) # ipgen_path points to the build dir
        self.set_nodeattr("fpgapart", fpgapart)
        self.set_nodeattr("clk_ns", str(clk_ns))

        # --- Call HLS Generation ---
        self.ipgen_singlenode_code(build_dir) # This runs HLS and unzips

        # --- Set final ip_path ---
        # ip_path should point to the *actual IP directory* containing component.xml
        final_exported_ip_dir = os.path.join(build_dir, "ip")
        self.set_nodeattr("ip_path", final_exported_ip_dir)

        print(f"✅ HLS project generation complete for {node.name} at {build_dir}")


    # --- IPI Generation (Block Design Instantiation) ---
    # This part needs careful review to ensure it uses the generated IP correctly.
    # The main change is using the VLNV correctly.

    def code_generation_ipi(self):
        """Generates Tcl commands for Vivado IPI block design instantiation."""
        node = self.onnx_node
        ip_instance_name = node.name # Instance name in the BD
        # ip_path points to the unzipped IP directory (e.g., build_dir/ip)
        ip_path = self.get_nodeattr("ip_path")
        # module_name is the HLS top function name used to generate the IP
        module_name = self.get_nodeattr("module_name")

        # --- Check IP Path ---
        if not ip_path or not os.path.isdir(ip_path) or not os.path.isfile(os.path.join(ip_path, "component.xml")):
            raise ValueError(
                f"ip_path attribute '{ip_path}' is not set, not a valid directory,"
                f" or does not contain component.xml for node {ip_instance_name}"
            )
        print(f"Using IP Path for IPI: {ip_path}") # Debug print

        # --- Determine VLNV ---
        # Default HLS export VLNV is xilinx.com:hls:<top_function_name>:<version>
        # Version is typically 1.0 unless specified otherwise in HLS project settings.
        vlnv = f"xilinx.com:hls:{module_name}:1.0"
        print(f"Instantiating IP with VLNV: {vlnv}") # Debug print

        # --- Generate Tcl Commands ---
        tcl_cmds = [
            f"puts \"<<<<< Executing IPI Tcl for CORRECTED StreamingSlice: {ip_instance_name} >>>>>\"",
            f"# Instantiating HLS IP {module_name} (VLNV: {vlnv}) for node {ip_instance_name}",
            # Use -vlnv argument for create_bd_cell when using IP Catalog IP
            f"create_bd_cell -type ip -vlnv {vlnv} {ip_instance_name}",
            f"puts \" Instantiated IP: {ip_instance_name} with VLNV {vlnv}\""
        ]

        # --- Apply standard FINN AXI Interface Properties ---
        # Get the interface names used in the BD (should match HLS port names)
        intf_names = self.get_verilog_top_module_intf_names() # Use the method that returns BD names like "in0", "out_r"

        s_axis_name = intf_names["s_axis"][0][0] # Should be "in0"
        m_axis_name = intf_names["m_axis"][0][0] # Should be "out_r"
        clk_pin_name = "ap_clk" # Default HLS clock pin name
        rst_pin_name = "ap_rst_n" # Default HLS reset pin name

        tcl_cmds += [
            f"# Set properties for IP Instance: {ip_instance_name}",
            # Associate interfaces to clock
            f"set_property -dict [list CONFIG.ASSOCIATED_BUSIF {{{s_axis_name}:{m_axis_name}}}] [get_bd_pins {ip_instance_name}/{clk_pin_name}]",
            # Associate reset to clock
            f"set_property -dict [list CONFIG.ASSOCIATED_RESET {{{rst_pin_name}}}] [get_bd_pins {ip_instance_name}/{clk_pin_name}]",
            # Set clock frequency (optional but good practice)
            # f"set_property -dict [list CONFIG.FREQ_HZ %d] [get_bd_pins {ip_instance_name}/{clk_pin_name}]" % round(1 / (float(self.get_nodeattr("clk_ns")) * 1e-9)),
            # Associate clock to interfaces
            f"set_property -dict [list CONFIG.ASSOCIATED_CLOCK {{{clk_pin_name}}}] [get_bd_intf_pins {ip_instance_name}/{s_axis_name}]",
            f"set_property -dict [list CONFIG.ASSOCIATED_CLOCK {{{clk_pin_name}}}] [get_bd_intf_pins {ip_instance_name}/{m_axis_name}]",
            # Associate reset to interfaces
            f"set_property -dict [list CONFIG.ASSOCIATED_RESET {{{rst_pin_name}}}] [get_bd_intf_pins {ip_instance_name}/{s_axis_name}]",
            f"set_property -dict [list CONFIG.ASSOCIATED_RESET {{{rst_pin_name}}}] [get_bd_intf_pins {ip_instance_name}/{m_axis_name}]",
             # Associate clock to reset pin
            f"set_property -dict [list CONFIG.ASSOCIATED_CLOCK {{{clk_pin_name}}}] [get_bd_pins {ip_instance_name}/{rst_pin_name}]",
            # Set reset polarity (optional, should be inferred)
            f"set_property -dict [list CONFIG.POLARITY {{ACTIVE_LOW}}] [get_bd_pins {ip_instance_name}/{rst_pin_name}]",
        ]
        tcl_cmds.append(f"puts \">>>>> Finished IPI Tcl for {ip_instance_name} <<<<<\"")

        return tcl_cmds

    def hls_sname(self, node_name=None):
        """Returns the default HLS module name, ensuring uniqueness."""
        if node_name is None:
             node_name = self.onnx_node.name
        # Sanitize node name to be valid C identifier / Verilog module name
        sanitized_name = "".join(c if c.isalnum() else "_" for c in node_name)
        # Ensure it starts with a letter or underscore
        if not sanitized_name or not (sanitized_name[0].isalpha() or sanitized_name[0] == '_'):
            sanitized_name = "hls_" + sanitized_name
        # Return a potentially unique name based on the ONNX node name
        return f"StreamingSlice_hls_{sanitized_name}" # Added _hls_ for clarity


    def get_verilog_top_module_intf_names(self):
        """Returns target interface names keyed by FINN interface type,
           using standard HLS AXI stream port names for IP Catalog."""
        return {
            "s_axis": [("in0", "in0")], # BD Intf Pin Name, (Internal wire name not needed for IPI)
            "m_axis": [("out_r", "out_r")],# BD Intf Pin Name
            "clk": ["ap_clk"],
            "rst": ["ap_rst_n"],
            "ap_none": [],
            "axilite": [],
            "aximm": [],
        }
# --- END OF FILE ---