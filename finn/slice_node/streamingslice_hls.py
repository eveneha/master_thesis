from finn.custom_op.fpgadataflow.streaming_slice import StreamingSlice

from finn.util.basic import make_build_dir
import os

class StreamingSlice_hls(StreamingSlice):
	"""HLS implementation for StreamingSlice. Inherits directly since no special HLS behavior is needed."""
	def ipgen_singlenode_code(self, build_dir):
		fpgapart = self.get_nodeattr("fpgapart")
		clk_ns = self.get_nodeattr("clk_ns")
		node = self.onnx_node
		node_name = node.name
		start_idx = self.get_nodeattr("start_idx")
		slice_length = self.get_nodeattr("slice_length")
		step = self.get_nodeattr("step")
		axis = self.get_nodeattr("axis")
		input_shape = self.get_nodeattr("input_shape")
		input_len = input_shape[axis]

		cpp_code = f"""#include "streaming_slice.hpp"

	extern "C" {{
	void StreamingSlice_top(hls::stream<ap_int<24>> &in0,
							hls::stream<ap_int<24>> &out) {{
		#pragma HLS interface axis port=in0
		#pragma HLS interface axis port=out
		#pragma HLS interface ap_ctrl_none port=return

		StreamingSlice<ap_int<24>, {input_len}, {slice_length}, {start_idx}, {step}>(in0, out);
	}}
	}}
	"""

		# Write source files
		cpp_path = os.path.join(build_dir, "streaming_slice.cpp")
		with open(cpp_path, "w") as f:
			f.write(cpp_code)

		hpp_src = os.path.join(os.path.dirname(__file__), "streaming_slice.hpp")
		hpp_dst = os.path.join(build_dir, "streaming_slice.hpp")
		os.system(f"cp {hpp_src} {hpp_dst}")

		# Write TCL script for Vitis HLS
		tcl_script = f"""open_project {build_dir}
	set_top StreamingSlice_top
	add_files streaming_slice.cpp
	add_files streaming_slice.hpp
	open_solution "solution1"
	set_part {fpgapart}
	create_clock -period {clk_ns} -name default
	csynth_design
 	config_export -disable_deadlock_detection 
	export_design -format ip_catalog -output {build_dir}/solution1/impl/ip
	exit
	"""

		tcl_path = os.path.join(build_dir, "run_hls.tcl")
		with open(tcl_path, "w") as f:
			f.write(tcl_script)

		# Run it
		ret = os.system(f"cd {build_dir} && vitis_hls -f run_hls.tcl")
		assert ret == 0, f"❌ Vivado HLS failed for {node_name}"

		print(f"✅ Generated StreamingSlice HLS IP and .xci at {build_dir}")




  
	def code_generation_ipgen(self, model, fpgapart, clk_ns):
		# Create build dir ONCE
		build_dir = make_build_dir(prefix="code_gen_ipgen_" + self.onnx_node.name)

		# Set build-related attributes BEFORE calling ipgen
		self.set_nodeattr("code_gen_dir_ipgen", build_dir)
		self.set_nodeattr("ipgen_path", build_dir)
		self.set_nodeattr("fpgapart", fpgapart)
		self.set_nodeattr("clk_ns", str(clk_ns))

		# Pass build_dir explicitly
		self.ipgen_singlenode_code(build_dir)

		# Post-HLS: set ip_path
		#self.set_nodeattr("ip_path", os.path.join(build_dir, "solution1", "impl", "ip", "hdl"))  
  
		# Set ip_path AFTER successful HLS
		ip_path = os.path.join(build_dir, "solution1", "impl", "ip", "hdl")
		assert os.path.isdir(ip_path), f"Expected HDL output folder not found: {ip_path}"
		self.set_nodeattr("ip_path", ip_path)
		print(f"✅ Generated StreamingSlice HLS project at {build_dir}")


	# def code_generation_ipi(self):
	# 	ip_name = self.onnx_node.name
	# 	ip_dir = self.get_nodeattr("code_gen_dir_ipgen")
	# 	ip_repo = os.path.join(ip_dir, "solution1", "impl")
	# 	ip_xci_path = os.path.join(ip_repo, "ip", "StreamingSlice_top.xci")

	# 	self.set_nodeattr("ip_path", os.path.join(ip_repo, "ip", "hdl"))

	# 	return [
	# 		f"set_property ip_repo_paths {{{os.path.abspath(ip_repo)}}} [current_project]",
	# 		"update_ip_catalog -rebuild",

	# 		# ✅ Step 1: Create .xci in Vivado internal folder
	# 		f"create_ip -name StreamingSlice_top -vendor xilinx.com -library hls -version 1.0 -module_name {ip_name}",
	# 		f"generate_target all [get_ips {ip_name}]",

	# 		# ✅ Step 2: Copy it out to the desired FINN path
	# 		f"file mkdir {{{os.path.dirname(ip_xci_path)}}}",
	# 		f"file copy -force [get_property IP_FILE [get_ips {ip_name}]] {{{ip_xci_path}}}",

	# 		# ✅ Step 3: Register it manually back
	# 		f"add_files -norecurse {{{ip_xci_path}}}",
	# 		"update_ip_catalog",

	# 		# ✅ Step 4: Use it in BD without re-creating it
	# 		f"create_bd_cell -type ip -vlnv xilinx.com:hls:StreamingSlice_top:1.0 {ip_name}",
	# 		f"make_bd_pins_external [get_bd_pins {ip_name}/ap_clk]",
	# 		f"make_bd_pins_external [get_bd_pins {ip_name}/ap_rst_n]",
	# 	]






#USES VERILOG RTL FILES:
	def code_generation_ipi(self):
		ip_name = self.onnx_node.name
		ip_dir = self.get_nodeattr("code_gen_dir_ipgen")
		hdl_path = os.path.join(ip_dir, "solution1", "impl", "ip", "hdl")#, "verilog")

		# vh_files = [
		# 	"StreamingSlice_top_hls_deadlock_kernel_monitor_top.vh"
		# ]

		# v_files = [
		# 	"StreamingSlice_top_flow_control_loop_pipe_sequential_init.v",
		# 	#"StreamingSlice_top_hls_deadlock_idx0_monitor.v",
		# 	#"StreamingSlice_top_hls_deadlock_idx1_monitor.v",
		# 	"StreamingSlice_top.v"
		# ]
  
		if not hdl_path or not os.path.isdir(hdl_path):
			raise ValueError(f"ip_path attribute '{hdl_path}' is not set or not a valid directory for node {ip_name}")

		verilog_path = os.path.join(hdl_path, "verilog")
		if not os.path.isdir(verilog_path):
			raise FileNotFoundError(f"Verilog subdirectory not found in {hdl_path}")

		# *** Dynamically find all .v and .vh files ***
		all_vh_files = [f for f in os.listdir(verilog_path) if f.endswith(".vh")]
		all_v_files = [f for f in os.listdir(verilog_path) if f.endswith(".v")]

		if not all_v_files:
			raise FileNotFoundError(f"No .v files found in {verilog_path}")

		tcl_cmds = []

		# ✅ Add .vh with read_verilog to mark as header-only
		# for f in vh_files:
		# 	tcl_cmds.append(f"read_verilog {os.path.join(hdl_path, f)}")

		# ✅ Add .v normally
		# for f in v_files:
		# 	tcl_cmds.append(f"add_files -norecurse {os.path.join(hdl_path, f)}")
		for f in all_vh_files:
			tcl_cmds.append(f"read_verilog {os.path.join(verilog_path, f)}") # Use read_verilog for headers
		for f in all_v_files:
			tcl_cmds.append(f"add_files -norecurse {os.path.join(verilog_path, f)}")
   
   
		# Create the BD cell
		tcl_cmds += [
			f"create_bd_cell -type module -reference StreamingSlice_top {ip_name}",
			f"puts \" Functions called by streaming_slice_HLS\"",
		]
  
		top_module_name = self.hls_sname()
		ip_name = self.onnx_node.name # Use the instance name

		tcl_cmds += [
			#f"create_bd_cell -type module -reference {top_module_name} {ip_name}",
			#f"puts \" Instantiated StreamingSlice HLS module: {ip_name}\"",
			# Add clock associations for the inferred interfaces
			f"set_property -dict [list CONFIG.ASSOCIATED_BUSIF {{in0:out_r}}] [get_bd_pins {ip_name}/ap_clk]",
			f"set_property -dict [list CONFIG.ASSOCIATED_RESET {{ap_rst_n}}] [get_bd_pins {ip_name}/ap_clk]",
			f"set_property -dict [list CONFIG.FREQ_HZ %d] [get_bd_pins {ip_name}/ap_clk]" % round(1 / (float(self.get_nodeattr("clk_ns")) * 1e-9)),
			# Explicitly associate reset to clock if needed (often inferred, but good practice)
			# f"set_property -dict [list CONFIG.ASSOCIATED_CLOCK {{ap_clk}}] [get_bd_pins {ip_name}/ap_rst_n]", # Optional
		]
		# Apply properties:
		# 1. Set properties on the CLOCK PIN (ap_clk)
		#    (Keep FREQ_HZ here, ASSOCIATED_BUSIF/RESET might be less effective but harmless)

		# Get AXI interface names
		s_axis_name = self.get_verilog_top_module_intf_names()["s_axis"][0][0] # "in0"
		m_axis_name = self.get_verilog_top_module_intf_names()["m_axis"][0][0] # "out_r"
  
		tcl_cmds += [
			f"set_property -dict [list CONFIG.ASSOCIATED_BUSIF {{{s_axis_name}:{m_axis_name}}}] [get_bd_pins {ip_name}/ap_clk]",
			f"set_property -dict [list CONFIG.ASSOCIATED_RESET {{ap_rst_n}}] [get_bd_pins {ip_name}/ap_clk]",
		]
		# 2. *** Set properties on the INTERFACE PINS (in0, out_r) *** <--- KEY CHANGE
		tcl_cmds += [
			# Associate s_axis 'in0' with clock and reset
			f"set_property -dict [list CONFIG.ASSOCIATED_CLOCK {{ap_clk}}] [get_bd_intf_pins {ip_name}/{s_axis_name}]",
			f"set_property -dict [list CONFIG.ASSOCIATED_RESET {{ap_rst_n}}] [get_bd_intf_pins {ip_name}/{s_axis_name}]",
			# Associate m_axis 'out_r' with clock and reset
			f"set_property -dict [list CONFIG.ASSOCIATED_CLOCK {{ap_clk}}] [get_bd_intf_pins {ip_name}/{m_axis_name}]",
			f"set_property -dict [list CONFIG.ASSOCIATED_RESET {{ap_rst_n}}] [get_bd_intf_pins {ip_name}/{m_axis_name}]",
		]
		# 3. Set property on the RESET PIN (ap_rst_n) - Ensure clock association here too
		tcl_cmds += [
				f"set_property -dict [list CONFIG.ASSOCIATED_CLOCK {{ap_clk}}] [get_bd_pins {ip_name}/ap_rst_n]",
				f"set_property -dict [list CONFIG.POLARITY {{ACTIVE_LOW}}] [get_bd_pins {ip_name}/ap_rst_n]", # Optional, should be inferred
		]
  
  
		return tcl_cmds

	def hls_sname(self):
		return "StreamingSlice_top"
 
	def get_verilog_top_module_intf_names(self):
		return {
			"s_axis": [("in0", self.hls_sname() + "_in0")], 
			"m_axis": [("out_r", self.hls_sname() + "_out_r")], 
			"clk": ["ap_clk"],
			"rst": ["ap_rst_n"],
			"ap_none": [],
			"axilite": [],
			"aximm": [],
		}





