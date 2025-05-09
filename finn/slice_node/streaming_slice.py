# streaming_slice.py

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.basic import make_build_dir
import os
from qonnx.core.datatype import DataType
import numpy as np

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
            "ip_path": ("s", False, ""),
            "module_name": ("s", True, ""),
            "num_channels": ("i", False, 8),
            "data_layout": ("s", False, "NCHW"),
        })
        return my_attrs



    def make_shape_compatible_op(self, model):
        import onnx.helper as oh

        start_idx = int(self.get_nodeattr("start_idx"))
        slice_length = int(self.get_nodeattr("slice_length"))
        axis = int(self.get_nodeattr("axis"))
        step = int(self.get_nodeattr("step"))
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
        print(f"🔁🔁🔁INFO: Inferring shape for StreamingSlice {self.onnx_node.name}")
        inp_node = self.onnx_node.input[0]
        inp_shape = model.get_tensor_shape(inp_node)
        assert inp_shape is not None, f"Input shape unknown for {self.onnx_node.name}!"
        if not inp_shape: return

        axis = self.get_nodeattr("axis")
        slice_len = self.get_nodeattr("slice_length")
        # --- Get data_layout, use default if not set ---
        data_layout = self.get_nodeattr("data_layout") # Will now get "NCHW" by default if not set

        print(f"DEBUG: {self.onnx_node.name} - inp_shape={inp_shape}, axis={axis}, data_layout='{data_layout}'") # Print layout

        out_shape = list(inp_shape)
        if axis < 0: axis += len(inp_shape)
        assert axis < len(inp_shape), f"Axis {axis} out of bounds for input shape {inp_shape}"
        out_shape[axis] = slice_len
        model.set_tensor_shape(self.onnx_node.output[0], out_shape)

        num_inner_elements = 1 # Default
        if len(inp_shape) == 4:
            if data_layout == "NHWC":
                print(f"DEBUG: {self.onnx_node.name} - Applying NHWC logic for num_channels")
                # Layout is [Batch, Height, Width, Channels]
                if axis == 0: # Slicing Batch (N)
                    num_inner_elements = inp_shape[1] * inp_shape[2] * inp_shape[3] # H*W*C
                elif axis == 1: # Slicing Height (H)
                    num_inner_elements = inp_shape[2] * inp_shape[3] # Interleaved elements are W*C
                elif axis == 2: # Slicing Width (W)
                    num_inner_elements = inp_shape[3] # Interleaved elements are C
                elif axis == 3: # Slicing Channels (C)
                    num_inner_elements = inp_shape[1] * inp_shape[2] # H*W (elements per channel slice)
                else:
                    warnings.warn(f"NHWC: Invalid axis {axis} for 4D shape.")
            elif data_layout == "NCHW": # Default NCHW if not NHWC and 4D
                print(f"DEBUG: {self.onnx_node.name} - Applying NCHW logic for num_channels")
                # Layout is [Batch, Channels, Height, Width]
                if axis == 0: # Slicing Batch (N)
                    num_inner_elements = inp_shape[1] * inp_shape[2] * inp_shape[3] # C*H*W
                elif axis == 1: # Slicing Channels (C)
                    num_inner_elements = inp_shape[2] * inp_shape[3] # H * W
                elif axis == 2: # Slicing Height (H)
                    num_inner_elements = inp_shape[1] * inp_shape[3] # C * W
                elif axis == 3: # Slicing Width (W)
                    num_inner_elements = inp_shape[1] # C
                else:
                    warnings.warn(f"NCHW: Invalid axis {axis} for 4D shape.")
            else:
                warnings.warn(f"Unrecognized data_layout '{data_layout}' for {self.onnx_node.name}. Defaulting num_channels logic.")
                # Fallback to a reasonable default if layout is unknown but 4D
                # This part is tricky without knowing what axis means for an unknown 4D layout.
                # For now, let's assume if axis is 2, C is at 1 and W is at 3 (like NCHW)
                if axis == 2: num_inner_elements = inp_shape[1] * inp_shape[3]
                elif axis == 1: num_inner_elements = inp_shape[2] * inp_shape[3]


        else: # Not 4D input
            warnings.warn(f"Input shape for {self.onnx_node.name} is not 4D ({len(inp_shape)}D). num_channels logic might be incorrect.")
            if axis < len(inp_shape) - 1:
                num_inner_elements = int(np.prod(inp_shape[axis+1:]))
            else:
                num_inner_elements = 1

        if num_inner_elements == 0 : num_inner_elements = 1
        self.set_nodeattr("num_channels", int(num_inner_elements))
        print(f"INFO: Inferred num_channels={int(num_inner_elements)} for StreamingSlice {self.onnx_node.name} (axis={axis}, input_shape={inp_shape}, data_layout='{data_layout}')")

        self.set_nodeattr("input_shape", inp_shape) # Store original input shape
        self.set_nodeattr("output_shape", out_shape) 

    def infer_node_datatype(self, model):
        assert model.get_tensor_datatype(self.onnx_node.input[0]) is not None, "Input dtype unknown!"
        inp_dtype = model.get_tensor_datatype(self.onnx_node.input[0])
        forced_dtype = "INT24" # 24-bit datatype
        finn_dtype = DataType[forced_dtype]
        model.set_tensor_datatype(self.onnx_node.output[0], finn_dtype)
        self.set_nodeattr("dataType", finn_dtype.name)

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
        return self.get_nodeattr("module_name")

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