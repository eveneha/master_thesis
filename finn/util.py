import onnx
import numpy as np
from finn.custom_op.fpgadataflow.streaming_slice import StreamingSlice

from qonnx.custom_op.registry import getCustomOp


# def transpose_and_rewire_between_transposes(model, start_tensor_name, end_tensor_name):
#     """
#     Absorb two known Transpose nodes doing NCHW->NHWC.
#     Modify nodes between by updating slicing axis and tensor shapes.
#     """

#     # Find Transpose nodes by tensor name
#     start_transpose = None
#     end_transpose = None
#     for node in model.graph.node:
#         if node.op_type == "Transpose":
#             if node.output[0] == start_tensor_name:
#                 start_transpose = node
#             if node.output[0] == end_tensor_name:
#                 end_transpose = node

#     if start_transpose is None or end_transpose is None:
#         raise Exception(f"Could not find transpose nodes with given tensor outputs {start_tensor_name} and {end_tensor_name}.")

#     print(f"🚀 Starting transpose absorption between {start_transpose.output[0]} and {end_transpose.output[0]}")

#     # Fixed permutation
#     perm = [0,2,3,1]      # NCHW -> NHWC
#     inverse_perm = [0,3,1,2]  # NHWC -> NCHW

#     # Find all nodes between start and end
#     start_output = start_transpose.output[0]
#     end_input = end_transpose.input[0]

#     queue = [start_output]
#     visited = set()
#     nodes_between = []

#     while queue:
#         tensor = queue.pop()
#         consumers = model.find_consumers(tensor)
#         for cons in consumers:
#             if cons.output[0] == end_transpose.output[0]:
#                 continue
#             if cons.name in visited:
#                 continue
#             visited.add(cons.name)
#             nodes_between.append(cons)
#             for out_tensor in cons.output:
#                 queue.append(out_tensor)

#     print(f"🧩 Nodes between transposes: {[n.op_type for n in nodes_between]}")

#     # Modify nodes between
#     for node in nodes_between:
#         if node.op_type == "StreamingSlice":
#             ss = StreamingSlice(node)
#             old_axis = ss.get_nodeattr("axis")
#             new_axis = inverse_perm[old_axis]
#             print(f"📐 StreamingSlice {node.name}: axis {old_axis} -> {new_axis}")
#             ss.set_nodeattr("axis", new_axis)

#         elif node.op_type == "MultiThreshold":
#             print(f"📐 MultiThreshold {node.name}: no change needed.")
#         elif node.op_type in ["MultiThreshold", "Thresholding"]:
#             print(f"📐 {node.op_type} {node.name}: updating to new axis")
#             # Find the axis attribute
#             for attr in node.attribute:
#                 if attr.name == "axis":
#                     print(f"📐 Old axis: {attr.i}")
#                     attr.i = 3
#                     print(f"📐 New axis: {attr.i}")

#         else:
#             print(f"⚠️ Node {node.name} of type {node.op_type} not explicitly handled.")

#     # Update output tensor shapes
#     for node in nodes_between:
#         for output_name in node.output:
#             if model.get_tensor_shape(output_name) is not None:
#                 old_shape = model.get_tensor_shape(output_name)
#                 if len(old_shape) == 4:
#                     new_shape = [old_shape[i] for i in perm]
#                     model.set_tensor_shape(output_name, new_shape)
#                     print(f"📐 Output {output_name} shape updated: {old_shape} -> {new_shape}")
#                 else:
#                     print(f"📐 Skipping non-4D output {output_name} with shape {old_shape}")

#     # Also update intermediate value_info shapes
#     for vi in model.graph.value_info:
#         if vi.name in visited:
#             old_shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
#             if len(old_shape) == 4:
#                 new_shape = [old_shape[i] for i in perm]
#                 for idx, val in enumerate(new_shape):
#                     vi.type.tensor_type.shape.dim[idx].dim_value = val
#                 print(f"📐 ValueInfo {vi.name} shape updated: {old_shape} -> {new_shape}")
#             else:
#                 print(f"📐 Skipping non-4D ValueInfo {vi.name} with shape {old_shape}")

#     # Remove the Transpose nodes
#     model.graph.node.remove(start_transpose)
#     model.graph.node.remove(end_transpose)

#     # Reconnect graph
#     for node in model.find_consumers(start_transpose.output[0]):
#         for idx, inp in enumerate(node.input):
#             if inp == start_transpose.output[0]:
#                 node.input[idx] = start_transpose.input[0]

#     for node in model.graph.node:
#         for idx, inp in enumerate(node.input):
#             if inp == end_transpose.output[0]:
#                 node.input[idx] = end_transpose.input[0]

#     print(f"✅ Transpose absorption complete.")

#     return model

import onnx
import numpy as np

def convert_node_output_to_nhwc(model, node_name):
    """
    Given a node name, permute its output tensor(s) from NCHW to NHWC format.
    Only changes tensor shapes, and updates fpgadataflow attributes if needed.
    """

    # Find the node by name
    target_node = None
    for node in model.graph.node:
        if node.name == node_name:
            target_node = node
            break

    if target_node is None:
        raise Exception(f"❌ Node with name {node_name} not found.")

    print(f"🚀 Found node {target_node.name} ({target_node.op_type}) to convert output tensor(s) to NHWC.")

    perm = [0,2,3,1]  # NCHW -> NHWC
    new_normal_shape = None

    # Update output tensor shapes (normal shapes)
    for output_name in target_node.output:
        if model.get_tensor_shape(output_name) is not None:
            old_shape = model.get_tensor_shape(output_name)
            if len(old_shape) == 4:
                new_normal_shape = [old_shape[i] for i in perm]
                model.set_tensor_shape(output_name, new_normal_shape)
                print(f"📐 Updated output {output_name} normal shape from {old_shape} to {new_normal_shape}")
            else:
                print(f"📐 Skipping non-4D output {output_name} with shape {old_shape}")

    # Only for fpgadataflow nodes
    if target_node.domain == "finn.custom_op.fpgadataflow" and new_normal_shape is not None:
        inst = getCustomOp(target_node)

        # Update normal_shape and output_shape
        inst.set_nodeattr("normal_shape", new_normal_shape)
        inst.set_nodeattr("output_shape", new_normal_shape)

        # Recompute folded_shape based on new normal_shape
        folded_shape = inst.get_nodeattr("folded_shape")
        if len(folded_shape) == 5:
            pe = folded_shape[-1]  # PE: folding parallelism
            new_folded_shape = new_normal_shape.copy()

            if len(new_folded_shape) == 4:
                # Insert extra 1 for folding
                # From (N, H, W, C) -> (N, H, W, OUTER_C, INNER_C)
                c = new_folded_shape[-1]
                assert c % pe == 0, f"Channel {c} not divisible by PE {pe}"
                new_outer_c = c // pe
                new_folded_shape = new_folded_shape[:-1] + [new_outer_c, pe]

            inst.set_nodeattr("folded_shape", new_folded_shape)
            print(f"📐 Updated folded shape {folded_shape} -> {new_folded_shape}")
        else:
            print(f"📐 Skipping folded_shape update for non-5D tensor {folded_shape}")

    return model


def fix_streamingdwchapes(model):
    """
    Fix StreamingDataWidthConverter nodes if their input/output shapes mismatch.
    Mainly adds missing dimension if needed.
    """

    for node in model.graph.node:
        if node.op_type == "StreamingDataWidthConverter":
            print(f"🔧 Fixing StreamingDataWidthConverter node: {node.name}")

            # Try to get folded_shape if exists
            try:
                inst = getCustomOp(node)
                folded_shape = inst.get_nodeattr("folded_shape")
                print(f"📐 folded_shape exists: {folded_shape}")
                continue  # If folded_shape exists, no problem, skip
            except Exception:
                print(f"⚠️ No folded_shape, fixing tensor shapes manually")

            # Fix output tensor shape
            for out_name in node.output:
                print(f"🔧 Checking output tensor {out_name}")
                if model.get_tensor_shape(out_name) is not None:
                    old_shape = model.get_tensor_shape(out_name)
                    if len(old_shape) == 4:
                        new_shape = old_shape + [1]
                        model.set_tensor_shape(out_name, new_shape)
                        print(f"📐 Fixed output {out_name} shape: {old_shape} -> {new_shape}")

            # Optionally fix input tensor shape too
            for in_name in node.input:
                if model.get_tensor_shape(in_name) is not None:
                    old_shape = model.get_tensor_shape(in_name)
                    if len(old_shape) == 4:
                        new_shape = old_shape + [1]
                        model.set_tensor_shape(in_name, new_shape)
                        print(f"📐 Fixed input {in_name} shape: {old_shape} -> {new_shape}")

    return model




def remove_node_and_rewire(model, node_name):
    """
    Remove a node by name and reconnect its input to its output consumers.
    """

    # Find the node
    target_node = None
    for node in model.graph.node:
        if node.name == node_name:
            target_node = node
            break

    if target_node is None:
        raise Exception(f"❌ Node with name {node_name} not found.")

    print(f"🚀 Found node {target_node.name} ({target_node.op_type}) to remove and rewire.")

    if len(target_node.input) != 1 or len(target_node.output) != 1:
        raise Exception(f"❌ Node {node_name} must have exactly 1 input and 1 output to rewire cleanly.")

    input_tensor = target_node.input[0]
    output_tensor = target_node.output[0]

    # Find consumers of the node's output
    consumers = model.find_consumers(output_tensor)

    for consumer in consumers:
        for idx, inp in enumerate(consumer.input):
            if inp == output_tensor:
                consumer.input[idx] = input_tensor
                print(f"🔗 Rewired {consumer.name}: input {idx} now from {input_tensor}")

    # Remove the node
    model.graph.node.remove(target_node)
    print(f"🗑️ Removed node {target_node.name}")

    return model


def update_node_attribute(model, node_name, attribute_name, new_value):
    """
    Update a node's attribute by name with a new value.
    """

    # Find the node
    target_node = None
    for node in model.graph.node:
        if node.name == node_name:
            target_node = node
            break

    if target_node is None:
        raise Exception(f"❌ Node with name {node_name} not found.")

    print(f"🚀 Found node {target_node.name} ({target_node.op_type}) to update attribute {attribute_name}.")

    # Find and update the attribute
    attr_found = False
    for attr in target_node.attribute:
        if attr.name == attribute_name:
            attr_found = True
            if isinstance(new_value, int):
                attr.i = new_value
                print(f"🛠️ Updated attribute {attribute_name}: {attr.i}")
            elif isinstance(new_value, float):
                attr.f = new_value
                print(f"🛠️ Updated attribute {attribute_name}: {attr.f}")
            elif isinstance(new_value, str):
                attr.s = new_value.encode("utf-8")
                print(f"🛠️ Updated attribute {attribute_name}: {attr.s}")
            elif isinstance(new_value, list):
                if all(isinstance(x, int) for x in new_value):
                    attr.ints[:] = new_value
                    print(f"🛠️ Updated attribute {attribute_name}: {attr.ints}")
                elif all(isinstance(x, float) for x in new_value):
                    attr.floats[:] = new_value
                    print(f"🛠️ Updated attribute {attribute_name}: {attr.floats}")
                else:
                    raise Exception(f"❌ List type not fully int or float!")
            else:
                raise Exception(f"❌ Unsupported new_value type: {type(new_value)}")
            break

    if not attr_found:
        raise Exception(f"❌ Attribute {attribute_name} not found in node {node_name}")

    return model
