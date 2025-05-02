# streaming_slice.py

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.basic import make_build_dir
import os
from qonnx.core.datatype import DataType


class StreamingSlice(HWCustomOp):
    def get_nodeattr_types(self):
        my_attrs = super().get_nodeattr_types()
        my_attrs.update({
            "start_idx": ("i", True, 0),
            "slice_length": ("i", True, 0),
            "axis": ("i", True, 0),
            "step": ("i", False, 1),
            "normal_shape": ("ints", True, []),
            "folded_shape": ("ints", True, []),
            "input_shape": ("ints", True, []),
            "output_shape": ("ints", True, []),
            "impl_style": ("s", True, "hls"),
            "dataType": ("s", True, ""), 
            "inFIFODepths": ("ints", False, [0]),
            "outFIFODepths": ("ints", False, [0]),
            "backend": ("s", True, "fpgadataflow"),
            "fpgapart": ("s", False, ""),      
            "clk_ns": ("s", False, "10"), 
            "ip_path": ("s", False, "")          
        })
        return my_attrs



    def make_shape_compatible_op(self, model):
        import onnx.helper as oh

        start_idx = int(self.get_nodeattr("start_idx"))
        slice_length = int(self.get_nodeattr("slice_length"))
        axis = int(self.get_nodeattr("axis"))
        step = int(self.get_nodeattr("step"))

        #print(f"🚨 DEBUG: start_idx={start_idx}, slice_length={slice_length}, axis={axis}, step={step}")

        starts = [start_idx]
        ends = [start_idx + slice_length * step]
        axes = [axis]
        steps = [step]

        new_node = oh.make_node(
            "Slice",
            inputs=[self.onnx_node.input[0]],
            outputs=[self.onnx_node.output[0]],
            name=self.onnx_node.name,
            starts=starts,
            ends=ends,
            axes=axes,
            steps=steps
        )
        return new_node



    def infer_node_shape(self, model):
        assert model.get_tensor_shape(self.onnx_node.input[0]) is not None, "Input shape unknown!"
        inp_shape = model.get_tensor_shape(self.onnx_node.input[0])
        axis = self.get_nodeattr("axis")
        slice_len = self.get_nodeattr("slice_length")
        out_shape = inp_shape.copy()
        out_shape[axis] = slice_len
        model.set_tensor_shape(self.onnx_node.output[0], out_shape)
        
        # Save shapes into node attributes
        self.set_nodeattr("input_shape", inp_shape)
        self.set_nodeattr("output_shape", out_shape)

    def infer_node_datatype(self, model):
        assert model.get_tensor_datatype(self.onnx_node.input[0]) is not None, "Input dtype unknown!"
        inp_dtype = model.get_tensor_datatype(self.onnx_node.input[0])
        model.set_tensor_datatype(self.onnx_node.output[0], inp_dtype)
        self.set_nodeattr("dataType", inp_dtype.name)

    def execute_node(self, context, graph):
        inp = context[self.onnx_node.input[0]]
        #inp = inp.permute(0, 2, 3, 1)  # NCHW -> NHWC
        axis = self.get_nodeattr("axis")
        start = self.get_nodeattr("start_idx")
        step = self.get_nodeattr("step")
        end = start + self.get_nodeattr("slice_length") * step
        slices = [slice(None)] * inp.ndim
        slices[axis] = slice(start, end, step)
        context[self.onnx_node.output[0]] = inp[tuple(slices)]

    def verify_node(self):
        assert self.get_nodeattr("start_idx") >= 0, "start_idx must be >= 0"
        assert self.get_nodeattr("slice_length") > 0, "slice_length must be > 0"
        assert self.get_nodeattr("step") > 0, "step must be > 0"

    # def code_generation_ipgen(self, model, fpgapart, clk):
    #     # wrapper to support PrepareIP
    #     self.ipgen_singlenode_code()

    # Hardware methods are not implemented yet!
    # def ipgen_singlenode_code(self):
        

    #     node = self.onnx_node
    #     node_name = node.name
    #     build_dir = make_build_dir(prefix="ipgen_" + node_name)

    #     # hardcode fpgapart and clk if needed (or retrieve later)
    #     fpgapart = "xc7z020clg400-1"
    #     clk = 10.0

    #     # Attributes from the ONNX node
    #     start_idx = self.get_nodeattr("start_idx")
    #     slice_length = self.get_nodeattr("slice_length")
    #     step = self.get_nodeattr("step")
    #     axis = self.get_nodeattr("axis")

    #     input_shape = self.get_nodeattr("input_shape")
    #     input_len = input_shape[axis]

    #     cpp_code = f"""#include "streaming_slice.hpp"

    #     extern "C" {{
    #     void StreamingSlice_top(hls::stream<ap_int<8>> &in0, hls::stream<ap_int<8>> &out) {{
    #     #pragma HLS interface axis port=in0
    #     #pragma HLS interface axis port=out
    #     #pragma HLS interface ap_ctrl_none port=return
    #         StreamingSlice<ap_int<8>, {input_len}, {slice_length}, {start_idx}, {step}>(in0, out);
    #     }}
    #     }}
    #     """

    #     cpp_path = os.path.join(build_dir, "streaming_slice.cpp")
    #     with open(cpp_path, "w") as f:
    #         f.write(cpp_code)

    #     header_src = os.path.join(os.path.dirname(__file__), "streaming_slice.hpp")
    #     header_dst = os.path.join(build_dir, "streaming_slice.hpp")
    #     os.system(f"cp {header_src} {header_dst}")

    #     tcl_script = f"""open_project {build_dir}
    #     set_top StreamingSlice_top
    #     add_files streaming_slice.cpp
    #     open_solution "sol1"
    #     set_part {fpgapart}
    #     create_clock -period {clk} -name default
    #     csynth_design
    #     export_design -format ip_catalog
    #     exit
    #     """
    #     tcl_path = os.path.join(build_dir, "run_hls.tcl")
    #     with open(tcl_path, "w") as f:
    #         f.write(tcl_script)

    #     self.set_nodeattr("ip_path", build_dir)
    #     print(f"✅ Generated StreamingSlice HLS project at {build_dir}")

    


    def get_normal_input_shape(self, ind=0):
        ishape = self.get_nodeattr("input_shape")
        return ishape
    
    
    def get_folded_input_shape(self, ind=0):
        # Calculate based on the normal *input* shape
        normal_ishape = self.get_normal_input_shape(ind)
        # Assuming SIMD=1 / PE=1 for the slice itself
        # (i.e., it processes one stream element at a time)
        if normal_ishape and len(normal_ishape) > 0:
            # Append the parallelism dimension (usually 1 for simple streaming ops)
            # Ensure this format matches what InsertFIFO expects (e.g., NHWC -> NHWCC')
            if len(normal_ishape) == 4: # NHWC
                return normal_ishape + [1] # Example folding for SIMD=1
            else:
                # Handle other dimensionalities if necessary
                return normal_ishape # Or raise error if unexpected dim
        else:
            warnings.warn(f"Could not determine normal input shape for {self.onnx_node.name}")
            return [] # Return empty list to indicate failure
    
    
    def get_folded_output_shape(self, ind=0):
        return self.get_nodeattr("folded_shape")


    def get_input_datatype(self, ind=0):
        return DataType[self.get_nodeattr("dataType")]

    def get_output_datatype(self, ind=0):
        return DataType[self.get_nodeattr("dataType")]

    def get_normal_output_shape(self, ind=0):
        oshape = self.get_nodeattr("output_shape")
        return oshape
    
    
    def get_folded_output_shape(self, ind=0):
        return self.get_nodeattr("folded_shape")




    def global_includes(self):
        return ["streaming_slice.hpp"]

    def get_verilog_top_module_name(self):
        return "StreamingSlice"

    def get_number_output_values(self):
        return self.get_nodeattr("slice_length")
    
    def get_outstream_width(self, ind=0):
        dtype = DataType[self.get_nodeattr("dataType")]
        folded_shape = self.get_nodeattr("folded_shape")
        out_width = folded_shape[-1] * dtype.bitwidth()
        return out_width


    def get_instream_width(self, ind=0):
        dtype = DataType[self.get_nodeattr("dataType")]
        folded_shape = self.get_nodeattr("folded_shape")
        in_width = folded_shape[-1] * dtype.bitwidth()
        return in_width
   

    

    

