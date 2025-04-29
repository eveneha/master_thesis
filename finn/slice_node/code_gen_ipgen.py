
import os 
import onnx 
from finn.util.basic import make_build_dir


def code_generation_ipgen(self, model, fpgapart, clk):
    import shutil

    # Get node
    node = self.onnx_node
    node_name = node.name

    # Build directory
    build_dir = make_build_dir(prefix="ipgen_{}".format(node_name))

    # Get slicing parameters
    start_idx = self.get_nodeattr("start_idx")
    slice_length = self.get_nodeattr("slice_length")
    axis = self.get_nodeattr("axis")
    step = self.get_nodeattr("step")

    # Assume fixed input datatype and axis = 2 (Time) for now
    dtype_str = "ap_int<8>"
    # Get input tensor shape
    in_shape = model.get_tensor_shape(node.input[0])
    num_in = in_shape[axis]
    num_out = slice_length

    # ---- Create .cpp file dynamically ----
    cpp_file = os.path.join(build_dir, "streaming_slice.cpp")
    with open(cpp_file, "w") as f:
        f.write(f"""#include "streaming_slice.hpp"
#include <hls_stream.h>
#include <ap_int.h>

extern "C" {{
void StreamingSlice_top(hls::stream<{dtype_str}> &in0, hls::stream<{dtype_str}> &out) {{
#pragma HLS INTERFACE axis port=in0
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE ap_ctrl_none port=return

StreamingSlice<{dtype_str}, {num_in}, {num_out}, {start_idx}, {step}>(in0, out);
}}
}}
""")

    # ---- Copy static .hpp file ----
    src_hpp = os.path.join(os.path.dirname(__file__), "streaming_slice.hpp")
    dst_hpp = os.path.join(build_dir, "streaming_slice.hpp")
    shutil.copyfile(src_hpp, dst_hpp)

    # ---- Create run_hls.tcl ----
    run_hls = os.path.join(build_dir, "run_hls.tcl")
    with open(run_hls, "w") as f:
        f.write(f"""open_project {build_dir}
set_top StreamingSlice_top
add_files streaming_slice.cpp
open_solution "sol1"
set_part {fpgapart}
create_clock -period {clk} -name default
csynth_design
export_design -format ip_catalog
exit
""")

    # Save build directory
    self.set_nodeattr("code_gen_dir_ipgen", build_dir)

    print(f"✅ HLS project generated at {build_dir}")
