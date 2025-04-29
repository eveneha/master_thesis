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
	void StreamingSlice_top(hls::stream<ap_int<8>> &in0, hls::stream<ap_int<8>> &out) {{
	#pragma HLS interface axis port=in0
	#pragma HLS interface axis port=out
	#pragma HLS interface ap_ctrl_none port=return
		StreamingSlice<ap_int<8>, {input_len}, {slice_length}, {start_idx}, {step}>(in0, out);
	}}
	}}
	"""
		with open(os.path.join(build_dir, "streaming_slice.cpp"), "w") as f:
			f.write(cpp_code)

		header_src = os.path.join(os.path.dirname(__file__), "streaming_slice.hpp")
		header_dst = os.path.join(build_dir, "streaming_slice.hpp")
		os.system(f"cp {header_src} {header_dst}")

		
		

		tcl_script = f"""open_project .
	set_top StreamingSlice_top
	add_files streaming_slice.cpp
	add_files streaming_slice.hpp
	open_solution "solution1"
	set_part {fpgapart}
	create_clock -period {clk_ns} -name default
	csynth_design
	export_design -format ip_catalog
	exit
	"""

		tcl_path = os.path.join(build_dir, "run_hls.tcl")
		with open(tcl_path, "w") as f:
			f.write(tcl_script)

		ret = os.system(f"cd {build_dir} && vitis_hls -f run_hls.tcl")
		assert ret == 0, f"❌ Vivado HLS failed for {node_name}"

		print(f"✅ Generated StreamingSlice HLS project at {build_dir}")


  
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


	def code_generation_ipi(self):
		ip_name = self.onnx_node.name
		ip_dir = self.get_nodeattr("code_gen_dir_ipgen")
		ip_path = os.path.join(ip_dir, "solution1", "impl", "ip", "hdl")

		self.set_nodeattr("ip_path", ip_path)

		tcl = f"""
	create_ip -name fifo_generator -vendor xilinx.com -library ip -version 13.2 -module_name {ip_name}
	set_property -dict [list CONFIG.INTERFACE_TYPE AXI_STREAM] [get_ips {ip_name}]
	set_property generate_synth_checkpoint false [get_files {ip_name}.xci]
	"""
 
 
 
 
		return tcl

	def hls_sname(self):
	    return "StreamingSlice_top"


