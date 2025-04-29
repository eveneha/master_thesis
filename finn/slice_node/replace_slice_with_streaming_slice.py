# file: replace_slice_with_streaming_slice.py

from qonnx.transformation.base import Transformation
from finn.custom_op.fpgadataflow.streaming_slice import StreamingSlice
import onnx

class ReplaceSliceWithStreamingSlice(Transformation):
    """Replace Slice with StreamingSlice, rewiring through Mul/Add chains."""

    def apply(self, model):
        graph = model.graph
        nodes_to_replace = []

        # Find all Slice nodes
        for node in graph.node:
            if node.op_type == "Slice" or node.op_type == "StreamingSlice":
                nodes_to_replace.append(node)

        for slice_node in nodes_to_replace:
            # Extract slicing parameters
            data_input = slice_node.input[0]
            data_output = slice_node.output[0]
            starts = model.get_initializer(slice_node.input[1])
            ends = model.get_initializer(slice_node.input[2])
            axes = model.get_initializer(slice_node.input[3])
            steps = model.get_initializer(slice_node.input[4])

            start_idx = int(starts[0])
            end_idx = int(ends[0])
            axis = int(axes[0])
            step = int(steps[0])

            #print(f"🚨 DEBUG: start_idx={start_idx}, end_idx={end_idx}, axis={axis}, step={step}")
            slice_len = (end_idx - start_idx + step - 1) // step

            # Create new StreamingSlice node
            ss_node = StreamingSlice(slice_node)
            ss_node.set_nodeattr("start_idx", start_idx)
            ss_node.set_nodeattr("slice_length", slice_len)
            ss_node.set_nodeattr("axis", axis)
            ss_node.set_nodeattr("step", step)
            ss_node.set_nodeattr("impl_style", "hls")
            ss_node.set_nodeattr("backend", "fpgadataflow")
            ss_node.set_nodeattr("fpgapart", "xc7z020clg400-1")  # or your actual target
            ss_node.set_nodeattr("clk_ns", "10.0")  # adjust clock period as needed

            # Infer input/output shapes
            input_shape = model.get_tensor_shape(data_input)
            print(f"📐📐📐📐📐 Original input shape: {input_shape}")
            output_shape = input_shape.copy()
            output_shape[axis] = slice_len
            print(f"📐📐📐📐📐 Updated output shape: {output_shape}")

            
            # Set normal/folded shape properly
            # Assume no folding except channel folding at the end
            normal_shape = output_shape
            # expand folded shape: make sure we split last axis
            folded_shape = output_shape.copy()
            if len(folded_shape) == 4:
                c = folded_shape[-1]
                folded_shape = folded_shape[:-1] + [1, c]  # split channel into (1, C)

            
            
            # Set normal/folded shape (assume no folding for now)
            ss_node.set_nodeattr("normal_shape", normal_shape)
            ss_node.set_nodeattr("folded_shape", folded_shape)
            print("📐📐📐📐📐 Set normal/folded shape: ", normal_shape, folded_shape)
            ss_node.set_nodeattr("input_shape", input_shape)
            ss_node.set_nodeattr("output_shape", output_shape)

            # Set data type
            inp_dtype = model.get_tensor_datatype(data_input)
            ss_node.set_nodeattr("dataType", inp_dtype.name)




            ss_node.onnx_node.op_type = "StreamingSlice"
            ss_node.onnx_node.domain = "finn.custom_op.fpgadataflow"

            current_input = data_input
            last_muladd = None
            chain_nodes = []

            # Walk backwards through Mul/Add chain
            while True:
                producer = model.find_producer(current_input)
                if producer is None or producer.op_type not in ["Mul", "Add"]:
                    break
                print(f"🔀 Skipping over {producer.op_type} node: {producer.name}")
                chain_nodes.append(producer)
                last_muladd = producer
                current_input = producer.input[0]

            if last_muladd is not None:
                # Create new tensor name for StreamingSlice output
                new_tensor_name = model.make_new_valueinfo_name()

                # Insert StreamingSlice
                ss_node.onnx_node.input[0] = current_input
                ss_node.onnx_node.output[0] = new_tensor_name

                # Rewire first Mul/Add to use StreamingSlice output
                first_node = chain_nodes[-1]
                for i in range(len(first_node.input)):
                    if first_node.input[i] == current_input:
                        first_node.input[i] = new_tensor_name

                # Rewire consumers of original Slice output to use last Mul/Add output
                for consumer in model.find_consumers(data_output):
                    for i in range(len(consumer.input)):
                        if consumer.input[i] == data_output:
                            consumer.input[i] = last_muladd.output[0]

                # --- Fix shape for Mul/Add nodes ---
                for node in chain_nodes:
                    out_tensor = node.output[0]
                   
                    if model.get_tensor_shape(out_tensor) is not None:
                        out_shape = model.get_tensor_shape(out_tensor)
                        print(f"📐 Original {node.op_type} output shape : {out_shape}")
                        axis = ss_node.get_nodeattr("axis")
                        slice_len = ss_node.get_nodeattr("slice_length")
                        out_shape[axis] = slice_len
                        model.set_tensor_shape(out_tensor, out_shape)
                        print(f"📐 Updated {node.op_type} output shape to {out_shape}")

            else:
                # No Mul/Add chain
                ss_node.onnx_node.input[0] = data_input
                ss_node.onnx_node.output[0] = data_output

            # --- Set StreamingSlice output shape ---
            slice_out_tensor = ss_node.onnx_node.output[0]
            if model.get_tensor_shape(slice_out_tensor) is None:
                input_shape = model.get_tensor_shape(ss_node.onnx_node.input[0])
                output_shape = input_shape.copy()
                axis = ss_node.get_nodeattr("axis")
                slice_len = ss_node.get_nodeattr("slice_length")
                output_shape[axis] = slice_len
                model.set_tensor_shape(slice_out_tensor, output_shape)
                print(f"📐 Set StreamingSlice output shape: {output_shape}")

            # --- 🔥 Fix downstream MultiThreshold or Thresholding ---
            for consumer in model.find_consumers(slice_out_tensor):
                if consumer.op_type in ["MultiThreshold", "Thresholding"]:
                    in_shape = model.get_tensor_shape(slice_out_tensor)
                    print(f"🛠️ Fixing {consumer.op_type} {consumer.name} input shape: {in_shape}")

                    # Very important: keep channels (axis 1) = 8
                    fixed_shape = in_shape.copy()
                    if len(fixed_shape) >= 2:
                        fixed_shape[1] = 8  # Channels stay 8
                    model.set_tensor_shape(consumer.input[0], fixed_shape)
                    model.set_tensor_shape(consumer.output[0], fixed_shape)

            # Remove old Slice and insert StreamingSlice
            graph.node.remove(slice_node)
            graph.node.append(ss_node.onnx_node)

        return (model, False)
