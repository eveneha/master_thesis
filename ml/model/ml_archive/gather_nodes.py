# --- START OF FILE prune_time_indices.py (Corrected Gather Insertion) ---

import numpy as np
import warnings
from copy import deepcopy
from functools import reduce
from typing import Dict, List, Tuple, Set
import math

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
#rom qonnx.util.basic import get_by_name, make_np_data_type_from_string
import onnx.helper as oh
from onnx import TensorProto

# --- Helper Functions ---
def calculate_output_length(input_length, kernel_size, dilation, stride):
     input_length = int(input_length); kernel_size = int(kernel_size); dilation = int(dilation); stride = int(stride)
     padding_start = 0; padding_end = 0
     output_len = math.floor((input_length + padding_start + padding_end - dilation * (kernel_size - 1) - 1) / stride + 1)
     return max(0, int(output_len))

class InsertTemporalGather(Transformation):
    """
    Inserts Gather nodes after specified tensors ONCE to select relevant temporal indices.
    Updates graph connections and shapes.
    """
    def __init__(self, relevant_indices_map: Dict[str, List[int]], temporal_axis=2, stop_node_name=None) -> None:
        super().__init__()
        self.relevant_indices_map = {}
        for tensor_name, indices in relevant_indices_map.items():
             try:
                  self.relevant_indices_map[tensor_name] = sorted(list(set(int(i) for i in indices)))
             except ValueError: warnings.warn(f"Invalid indices for {tensor_name}. Skipping.")
        self.temporal_axis = temporal_axis
        self.stop_node_name = stop_node_name
        # Keep track of tensors for which Gather has already been inserted across calls to apply
        self.gather_inserted_for = set()

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        print("--- InsertTemporalGather: Applying Gather nodes for temporal pruning ---")
        graph = model.graph
        nodes_topo_order = list(graph.node) # Get nodes in topological order
        graph_changed_this_pass = False # Track changes in *this* pass
        nodes_to_add = []
        initializers_to_add = []
        rewiring_map = {} # tensor_name -> gather_output_name

        # Determine tensors to skip based on stop_node_name
        tensors_to_skip = set()
        if self.stop_node_name:
            stop_node_found = False
            for node in nodes_topo_order:
                if stop_node_found:
                    tensors_to_skip.update(node.input); tensors_to_skip.update(node.output)
                elif node.name == self.stop_node_name:
                    stop_node_found = True
                    tensors_to_skip.update(node.input); tensors_to_skip.update(node.output)

        # Iterate through the map of tensors requiring gathering
        for tensor_name, relevant_indices in self.relevant_indices_map.items():

            # *** Check if already processed in a previous call or should be skipped ***
            if tensor_name in self.gather_inserted_for or tensor_name in tensors_to_skip:
                continue

            if not relevant_indices:
                continue

            # Check if tensor exists and get shape
            current_shape = model.get_tensor_shape(tensor_name)
            if current_shape is None or len(current_shape) <= self.temporal_axis or not isinstance(current_shape[self.temporal_axis], int):
                warnings.warn(f"Cannot insert Gather for {tensor_name}: incompatible shape {current_shape}.")
                continue

            current_temporal_size = current_shape[self.temporal_axis]
            valid_relevant_indices = [idx for idx in relevant_indices if 0 <= idx < current_temporal_size]

            if not valid_relevant_indices:
                 warnings.warn(f"No valid relevant indices for {tensor_name} (shape {current_shape}). Skipping.")
                 continue

            # *** If gather results in no change, skip insertion ***
            if len(valid_relevant_indices) == current_temporal_size:
                 # print(f"  Skipping Gather for {tensor_name}: all indices relevant.")
                 self.gather_inserted_for.add(tensor_name) # Mark as processed
                 continue

            # --- Create Gather Node and Constant Indices ---
            print(f"  Inserting Gather for tensor: {tensor_name} (keeping {len(valid_relevant_indices)} indices)")
            graph_changed_this_pass = True

            indices_tensor_name = model.make_new_valueinfo_name() # Use helper for unique name
            indices_array = np.array(valid_relevant_indices, dtype=np.int64)
            indices_tensor = oh.make_tensor( name=indices_tensor_name, data_type=TensorProto.INT64, dims=indices_array.shape, vals=indices_array.tobytes(), raw=True)
            initializers_to_add.append(indices_tensor)

            gather_node_name = tensor_name + "_gather_node"
            gather_output_name = model.make_new_valueinfo_name()
            gather_node = oh.make_node("Gather", inputs=[tensor_name, indices_tensor_name], outputs=[gather_output_name], name=gather_node_name, axis=self.temporal_axis)
            nodes_to_add.append(gather_node)

            # Update shape info for the Gather output
            new_temporal_size = len(valid_relevant_indices)
            new_shape = list(current_shape)
            new_shape[self.temporal_axis] = new_temporal_size
            model.set_tensor_shape(gather_output_name, tuple(new_shape))

            # Record the rewiring needed
            rewiring_map[tensor_name] = gather_output_name
            self.gather_inserted_for.add(tensor_name) # Mark as processed


        # --- Perform Rewiring After Identifying All Gather Nodes ---
        if rewiring_map:
            for node in graph.node:
                # Skip newly added Gather nodes themselves
                if node.name in [n.name for n in nodes_to_add]:
                     continue
                # Skip nodes after stop node
                if node.name in tensors_to_skip: # Reuse tensors_to_skip logic (approximate)
                     continue

                for i, inp_name in enumerate(node.input):
                    if inp_name in rewiring_map:
                         # print(f"    Rewiring input {i} of node {node.name} from {inp_name} to {rewiring_map[inp_name]}")
                         node.input[i] = rewiring_map[inp_name]

            # Check graph outputs
            for i, out_tensor in enumerate(graph.output):
                 if out_tensor.name in rewiring_map:
                      print(f"    Updating graph output from {out_tensor.name} to {rewiring_map[out_tensor.name]}")
                      graph.output[i].name = rewiring_map[out_tensor.name]


        # Add new nodes and initializers
        graph.node.extend(nodes_to_add)
        graph.initializer.extend(initializers_to_add)

        # Cleanup graph
        if graph_changed_this_pass:
            model.cleanup()

        print(f"--- InsertTemporalGather: Finished Pass. Inserted {len(nodes_to_add)} new Gather nodes this pass. ---")
        # Return False because this transform should complete in one pass now
        return (model, False)


# --- PruneSamples Class (Calls InsertTemporalGather) ---
# ... (Keep imports and InsertTemporalGather class as corrected before) ...

class PruneSamples(Transformation):
    """Top-level transformation to prune temporal dimension using Gather nodes."""

    def __init__(self, relevant_indices_map: Dict, lossy: bool = True, stop_node_name=None) -> None:
        super().__init__()
        self.relevant_indices_map = relevant_indices_map
        self.lossy = lossy # lossy flag currently unused
        self.stop_node_name = stop_node_name
        # Create instance of InsertTemporalGather internally
        # Pass self.relevant_indices_map and self.stop_node_name
        self.gather_inserter = InsertTemporalGather(
            relevant_indices_map=self.relevant_indices_map,
            stop_node_name=self.stop_node_name
        )

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:

        # Apply Gather Nodes based on relevant indices
        # Pass the internal gather_inserter instance to model.transform
        # *** FIX: Assign result directly without unpacking ***
        model_after_gather = model.transform(self.gather_inserter)

        # The gather_inserter's apply method now returns (model, False)
        # and tracks changes internally. The PruneSamples wrapper itself
        # doesn't need to signal further changes based on the gather step.
        return (model_after_gather, False) # Return the modified model and False

# --- END OF FILE ---