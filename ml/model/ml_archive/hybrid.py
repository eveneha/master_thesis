# --- START OF FILE prune_time_indices.py (Hybrid Approach) ---

import numpy as np
import warnings
from copy import deepcopy
from functools import reduce
from typing import Dict, List, Tuple, Set
import math

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
#from qonnx.util.basic import get_by_name, make_np_data_type_from_string
import onnx.helper as oh
from onnx import TensorProto

# --- Helper Functions ---
# (Keep ensure_masktype_is_dict_temporal, merge_dicts_of_sets_temporal, remove_masked_tensor_samples)
# Ensure remove_masked_tensor_samples is the version that handles shape tuples correctly
def ensure_masktype_is_dict_temporal(mask):
    # ... (implementation from previous answers) ...
    if isinstance(mask, dict): return mask
    if mask is None: return dict()
    else: raise Exception("Cannot turn %s into dict" % str(mask))

def merge_dicts_of_sets_temporal(dict1, dict2):
     # ... (implementation ensuring integer keys from previous answers) ...
    ret = deepcopy(dict1); final_ret = {}
    for key, val in dict2.items():
        try: int_key = int(key)
        except ValueError: int_key = key
        if int_key in ret.keys():
            set1 = set(ret[int_key]) if isinstance(ret[int_key], list) else ret[int_key]
            set2 = set(val) if isinstance(val, list) else val
            ret[int_key] = set1.union(set2)
        else: ret[int_key] = set(val) if isinstance(val, list) else val
    for k, v in ret.items():
        try: final_ret[int(k)] = v
        except ValueError: final_ret[k] = v
    return final_ret


def remove_masked_tensor_samples(tensor_or_shape, mask, axis):
    # ... (implementation from previous answers, ensuring it handles shape tuples) ...
    shape_only = False; mask_list = list(mask) if not isinstance(mask, list) else mask
    if type(tensor_or_shape) in [list, tuple]: shape_only = True; tensor_or_shape_np = np.zeros(tensor_or_shape) # Use dummy for shape calc
    else: tensor_or_shape_np = tensor_or_shape
    assert type(tensor_or_shape_np) is np.ndarray, f"Expected numpy array, got {type(tensor_or_shape_np)}"
    if axis >= tensor_or_shape_np.ndim or not mask_list: return tensor_or_shape
    if tensor_or_shape_np.ndim == 0 or np.prod(tensor_or_shape_np.shape) == 1: return tensor_or_shape
    else:
        current_dim_size = tensor_or_shape_np.shape[axis]
        valid_mask_list = sorted(list(set([int(idx) for idx in mask_list if isinstance(idx, (int, str)) and str(idx).isdigit() and 0 <= int(idx) < current_dim_size])))
        if not valid_mask_list: return tensor_or_shape
        if shape_only: new_shape = list(tensor_or_shape); new_shape[axis] -= len(valid_mask_list); ret = tuple(new_shape)
        else: ret = np.delete(tensor_or_shape_np, valid_mask_list, axis=axis)
    return ret


# --- Transformations ---

class ApplyTemporalMasks(Transformation):
    # ... (implementation from previous answers, ensures int keys/sets) ...
    def __init__(self, prune_spec: Dict) -> None: super().__init__(); self.prune_spec = prune_spec
    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        assert isinstance(self.prune_spec, dict)
        for key, val in self.prune_spec.items():
            assert isinstance(key, str); assert isinstance(val, dict)
            integer_axis_val = {}
            for k_str, v_list in val.items():
                 try: k_int = int(k_str); integer_axis_val[k_int] = set(int(idx) for idx in v_list if isinstance(idx, (int, str)) and str(idx).isdigit())
                 except ValueError: warnings.warn(f"Mask key '{k_str}' for {key} not integer. Ignoring.")
            if integer_axis_val: model.set_tensor_sparsity(key, integer_axis_val)
        return (model, False)


class RemoveMaskedSamples(Transformation):
    """(Simplified) Remove temporal samples based on initial masks in a single pass
       and SETS tensor shapes directly based on pruning."""
    def __init__(self, lossy: bool = True, stop_node_name=None) -> None:
        super().__init__()
        self.lossy = lossy
        self.stop_node_name = stop_node_name

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        print("--- RemoveMaskedSamples: Single Pass Removal & Shape Setting ---")
        model_current = model
        nodes_topo_order = list(model_current.graph.node)
        stop_reached = False
        any_change_made = False
        processed_tensors = set()
        temporal_axis = 2 # Assume axis 2

        tensors_to_skip = set()
        if self.stop_node_name:
            # Determine tensors associated with nodes at or after the stop node
            stop_node_found = False
            for node in nodes_topo_order:
                 if stop_node_found: tensors_to_skip.update(node.input); tensors_to_skip.update(node.output)
                 elif node.name == self.stop_node_name:
                     stop_node_found = True; tensors_to_skip.update(node.input); tensors_to_skip.update(node.output)

        all_tensor_names = set([t.name for t in model_current.graph.value_info] +
                               [t.name for t in model_current.graph.input] +
                               [t.name for t in model_current.graph.output] +
                               [t.name for t in model_current.graph.initializer])

        for tensor_name in all_tensor_names:
            if tensor_name in processed_tensors or tensor_name in tensors_to_skip: continue

            mask = model_current.get_tensor_sparsity(tensor_name)
            if mask is None or mask == {}: continue

            processed_tensors.add(tensor_name)
            temporal_mask_set = mask.get(temporal_axis)

            if temporal_mask_set is None or not temporal_mask_set:
                model_current.set_tensor_sparsity(tensor_name, {}) # Clear non-temporal
                continue

            temporal_mask_set = set(int(idx) for idx in temporal_mask_set if isinstance(idx, (int, str)) and str(idx).isdigit())

            initializer = model_current.get_initializer(tensor_name)
            io_t = initializer
            current_shape_tuple = model_current.get_tensor_shape(tensor_name)

            if current_shape_tuple is None or len(current_shape_tuple) <= temporal_axis or not isinstance(current_shape_tuple[temporal_axis], int):
                warnings.warn(f"Cannot apply mask to tensor {tensor_name} shape {current_shape_tuple}. Clearing mask.")
                model_current.set_tensor_sparsity(tensor_name, {})
                continue

            current_temporal_size = current_shape_tuple[temporal_axis]
            valid_discard_indices = set(idx for idx in temporal_mask_set if 0 <= idx < current_temporal_size)
            final_pruned_size = current_temporal_size - len(valid_discard_indices)

            if not valid_discard_indices and final_pruned_size == current_temporal_size:
                model_current.set_tensor_sparsity(tensor_name, {})
                continue

            new_shape = list(current_shape_tuple)
            new_shape[temporal_axis] = final_pruned_size
            new_shape_tuple = tuple(new_shape)

            if new_shape_tuple != current_shape_tuple:
                # print(f"  Setting shape for {tensor_name} from {current_shape_tuple} to {new_shape_tuple}")
                model_current.set_tensor_shape(tensor_name, new_shape_tuple)
                any_change_made = True

            if io_t is not None:
                if temporal_axis < io_t.ndim and io_t.shape[temporal_axis] > 0:
                    valid_indices_in_data = [idx for idx in valid_discard_indices if 0 <= idx < io_t.shape[temporal_axis]]
                    if valid_indices_in_data:
                         new_t = remove_masked_tensor_samples(io_t, valid_indices_in_data, axis=temporal_axis)
                         if new_t.shape != io_t.shape:
                             # print(f"  Pruning initializer {tensor_name} from {io_t.shape} to {new_t.shape}")
                             model_current.set_initializer(tensor_name, new_t)
                             any_change_made = True
                             if model_current.get_tensor_shape(tensor_name) != new_shape_tuple:
                                 warnings.warn(f"Shape mismatch for initializer {tensor_name}. Setting to {new_shape_tuple}")
                                 model_current.set_tensor_shape(tensor_name, new_shape_tuple)
                else:
                     if temporal_axis in mask:
                          warnings.warn(f"Temporal mask (axis {temporal_axis}) for initializer {tensor_name} shape {io_t.shape} ignored.")

            model_current.set_tensor_sparsity(tensor_name, {})

        print(f"--- RemoveMaskedSamples: Finished. Changes made: {any_change_made} ---")
        return (model_current, False) # Single pass complete


class InsertTemporalGather(Transformation):
    # ... (Keep the corrected implementation from the previous answer, including self.gather_inserted_for) ...
    def __init__(self, relevant_indices_map: Dict[str, List[int]], temporal_axis=2, stop_node_name=None) -> None:
        super().__init__(); self.temporal_axis = temporal_axis; self.stop_node_name = stop_node_name
        self.relevant_indices_map = {}
        for tensor_name, indices in relevant_indices_map.items():
             try: self.relevant_indices_map[tensor_name] = sorted(list(set(int(i) for i in indices)))
             except ValueError: warnings.warn(f"Invalid indices for {tensor_name}. Skipping.")
        self.gather_inserted_for = set() # Track insertions

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        print("--- InsertTemporalGather: Applying Gather nodes AFTER Conv ---")
        graph = model.graph; nodes_topo_order = list(graph.node); graph_changed_this_pass = False
        nodes_to_add = []; initializers_to_add = []; rewiring_map = {}

        tensors_to_skip = set()
        if self.stop_node_name:
            stop_node_found = False
            for node in nodes_topo_order:
                if stop_node_found: tensors_to_skip.update(node.input); tensors_to_skip.update(node.output)
                elif node.name == self.stop_node_name: stop_node_found = True; tensors_to_skip.update(node.input); tensors_to_skip.update(node.output)

        # Iterate ONLY through Conv node outputs specified in the map
        for tensor_name, relevant_indices in self.relevant_indices_map.items():
            if tensor_name in self.gather_inserted_for or tensor_name in tensors_to_skip: continue
            producer_node = model.find_producer(tensor_name)
            # *** IMPORTANT: Only insert Gather if the producer is a Conv node ***
            if not producer_node or producer_node.op_type not in ["Conv", "QuantConv"]: # Adjust op_types if needed
                continue

            if not relevant_indices: continue

            current_shape = model.get_tensor_shape(tensor_name)
            if current_shape is None or len(current_shape) <= self.temporal_axis or not isinstance(current_shape[self.temporal_axis], int):
                 warnings.warn(f"Cannot insert Gather for {tensor_name}: shape {current_shape}."); continue

            current_temporal_size = current_shape[self.temporal_axis]
            valid_relevant_indices = [idx for idx in relevant_indices if 0 <= idx < current_temporal_size]

            if not valid_relevant_indices: warnings.warn(f"No valid indices for {tensor_name}. Skipping."); continue
            if len(valid_relevant_indices) == current_temporal_size: self.gather_inserted_for.add(tensor_name); continue

            print(f"  Inserting Gather for tensor: {tensor_name} (Conv output, keeping {len(valid_relevant_indices)} indices)")
            graph_changed_this_pass = True

            indices_tensor_name = model.make_new_valueinfo_name()
            indices_array = np.array(valid_relevant_indices, dtype=np.int64)
            indices_tensor = oh.make_tensor(name=indices_tensor_name, data_type=TensorProto.INT64, dims=indices_array.shape, vals=indices_array.tobytes(), raw=True)
            initializers_to_add.append(indices_tensor)

            gather_node_name = tensor_name + "_gather_node" # Simple name
            gather_output_name = model.make_new_valueinfo_name()
            gather_node = oh.make_node("Gather", inputs=[tensor_name, indices_tensor_name], outputs=[gather_output_name], name=gather_node_name, axis=self.temporal_axis)
            nodes_to_add.append(gather_node)

            new_temporal_size = len(valid_relevant_indices)
            new_shape = list(current_shape); new_shape[self.temporal_axis] = new_temporal_size
            model.set_tensor_shape(gather_output_name, tuple(new_shape))

            rewiring_map[tensor_name] = gather_output_name
            self.gather_inserted_for.add(tensor_name)

        # Perform Rewiring
        if rewiring_map:
            for node in graph.node:
                if node.name in [n.name for n in nodes_to_add]: continue
                should_skip_node = False;
                for out_name in node.output:
                     if out_name in tensors_to_skip: should_skip_node = True; break
                if should_skip_node: continue
                for i, inp_name in enumerate(node.input):
                    if inp_name in rewiring_map: node.input[i] = rewiring_map[inp_name]
            for i, out_tensor in enumerate(graph.output):
                 if out_tensor.name in rewiring_map: graph.output[i].name = rewiring_map[out_tensor.name]

        graph.node.extend(nodes_to_add)
        graph.initializer.extend(initializers_to_add)
        if graph_changed_this_pass: model.cleanup()

        print(f"--- InsertTemporalGather: Finished. Inserted {len(nodes_to_add)} Gather nodes. ---")
        return (model, False) # Single pass


# --- New Hybrid Pruning Transformation ---
class PruneSamplesHybrid(Transformation):
    """Hybrid approach: Remove samples & set shapes first, then insert Gather after Conv."""

    def __init__(self, prune_spec: Dict, relevant_indices_conv_map: Dict, lossy: bool = True, stop_node_name=None) -> None:
        super().__init__()
        self.prune_spec = prune_spec # Based on non-relevant indices for *all* targeted tensors
        self.relevant_indices_conv_map = relevant_indices_conv_map # Based on relevant indices for *Conv outputs only*
        self.lossy = lossy
        self.stop_node_name = stop_node_name
        # Instantiate child transformations
        self.mask_applier = ApplyTemporalMasks(self.prune_spec)
        self.sample_remover = RemoveMaskedSamples(self.lossy, self.stop_node_name)
        self.gather_inserter = InsertTemporalGather(self.relevant_indices_conv_map, stop_node_name=self.stop_node_name)


    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        print("--- PruneSamplesHybrid: Starting Hybrid Pruning ---")

        # 1. Apply initial masks from prune_spec
        model, _ = model.transform(self.mask_applier)
        print("Step 1/3: ApplyTemporalMasks completed.")

        # 2. Remove samples and set shapes based on masks
        model, _ = model.transform(self.sample_remover)
        print("Step 2/3: RemoveMaskedSamples completed.")

        # 3. Insert Gather nodes specifically after Conv outputs
        model, _ = model.transform(self.gather_inserter)
        print("Step 3/3: InsertTemporalGather completed.")

        print("--- PruneSamplesHybrid: Finished Hybrid Pruning ---")
        # Overall transformation completes in these steps
        return (model, False)

# --- END OF FILE ---