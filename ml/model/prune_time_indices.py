import numpy as np
import warnings
from copy import deepcopy
from functools import reduce
from typing import Dict, Tuple

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name
import onnx.helper as oh # Added for shape inference


# (Keep the existing helper functions like ensure_masktype_is_dict_temporal, merge_dicts_of_sets_temporal, remove_masked_tensor_samples)


def ensure_masktype_is_dict_temporal(mask):
    if isinstance(mask, dict):
        # all good, return as is
        return mask
    if mask is None:
        # use empty dict instead of no sparsity mask (None)
        return dict()
    else:
        raise Exception("Cannot turn %s into dict" % str(mask))

def merge_dicts_of_sets_temporal(dict1, dict2):
    ret = deepcopy(dict1)
    for key, val in dict2.items():
        # Attempt to convert key to int, handle non-int keys gracefully
        try:
            int_key = int(key)
        except ValueError:
            int_key = key # Keep as is if not an integer string

        if int_key in ret.keys():
            # Convert to sets before update to handle potential lists
            set1 = set(ret[int_key]) if isinstance(ret[int_key], list) else ret[int_key]
            set2 = set(val) if isinstance(val, list) else val
            ret[int_key] = set1.union(set2) # Union of sets
        else:
            ret[int_key] = set(val) if isinstance(val, list) else val # Ensure value is a set

    # After merging, ensure all keys are integers where possible
    final_ret = {}
    for k, v in ret.items():
        try:
            final_ret[int(k)] = v
        except ValueError:
             warnings.warn(f"Non-integer key found in mask during merge: {k}. Keeping as is.")
             final_ret[k] = v # Keep non-integer keys if they exist
    return final_ret


def remove_masked_tensor_samples(tensor_or_shape, mask, axis):
    shape_only = False
    # Ensure mask is a list of indices for np.delete
    mask_list = list(mask) if not isinstance(mask, list) else mask

    if type(tensor_or_shape) in [list, tuple]:
        shape_only = True
        # Use a dummy array of the correct shape for shape calculation
        tensor_or_shape = np.zeros(tensor_or_shape) # Use zeros instead of random for deterministic shapes

    assert type(tensor_or_shape) is np.ndarray or shape_only, f"Expected numpy array or shape tuple/list, got {type(tensor_or_shape)}"
    if (not shape_only and (tensor_or_shape.ndim <= axis or tensor_or_shape.shape[axis] == 0)) or not mask_list:
        # No pruning if tensor has fewer dimensions than axis, dimension is empty, or no masks to apply
        # Or if it's shape_only and the specified axis is out of bounds
         if shape_only and axis >= len(tensor_or_shape):
             # warnings.warn(f"Cannot prune shape {tensor_or_shape} along axis {axis} - axis is out of bounds.") # Suppress this warning as it's checked later
             return tensor_or_shape # Return original shape if axis out of bounds
         elif not shape_only and (tensor_or_shape.ndim <= axis or tensor_or_shape.shape[axis] == 0):
             # warnings.warn(f"Cannot prune tensor with shape {tensor_or_shape.shape} along axis {axis} - axis out of bounds or dimension is zero.") # Suppress this
             return tensor_or_shape # Return original tensor if axis out of bounds or dim empty
         else:
             return tensor_or_shape # Return original if no mask list

    if not shape_only and (tensor_or_shape.ndim == 0 or np.prod(tensor_or_shape.shape) == 1):
        # no pruning for scalar properties
        return tensor_or_shape
    else:
        # Use current_dim_size for bound checks regardless of shape_only or not
        current_dim_size = tensor_or_shape[axis] if shape_only else tensor_or_shape.shape[axis]

        # Ensure mask indices are within the bounds of the axis dimension
        valid_mask_list = sorted(list(set([idx for idx in mask_list if 0 <= idx < current_dim_size])))

        if not valid_mask_list:
            return tensor_or_shape # No valid indices to delete

        if shape_only:
            # Manually calculate the new shape
            new_shape = list(tensor_or_shape)
            new_shape[axis] -= len(valid_mask_list)
            ret = tuple(new_shape)
        else:
            # Use np.delete on the numpy array
            # We already filtered valid_mask_list, so np.delete shouldn't raise IndexError
            ret = np.delete(tensor_or_shape, valid_mask_list, axis=axis)

    if shape_only:
        return ret
    else:
        return ret

# Removed update_node_mask_temporal
# Removed PropagateTemporalMasks

class ApplyTemporalMasks(Transformation):
    """Apply the given temporal sparsity masks in prune_spec to the appropriately named
    tensors in the model."""

    def __init__(self, prune_spec: Dict) -> None:
        super().__init__()
        self.prune_spec = prune_spec

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        # sanity check:
        assert isinstance(self.prune_spec, dict)
        for key, val in self.prune_spec.items():
            assert isinstance(key, str)
            assert isinstance(val, dict)
            # Ensure axis keys are integers when applying the initial mask
            # And ensure indices are sets
            integer_axis_val = {}
            for k_str, v_list in val.items():
                 try:
                     k_int = int(k_str)
                     integer_axis_val[k_int] = set(v_list) # Ensure indices are a set
                 except ValueError:
                     warnings.warn(f"Sparsity mask key '{k_str}' for tensor {key} in prune_spec is not an integer. Ignoring.")

            if integer_axis_val:
                model.set_tensor_sparsity(key, integer_axis_val)
        return (model, False)


class RemoveMaskedSamples(Transformation):
    """Remove temporal samples indicated by sparsity masks and update downstream shapes."""

    def __init__(self, lossy: bool = True, stop_node_name=None, max_iterations=100) -> None:
        super().__init__()
        self.lossy = lossy
        self.stop_node_name = stop_node_name
        self.max_iterations = max_iterations # Safeguard against infinite loops

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        model_current = model # Operate on the input model
        need_rerun_outer = False # Flag for the outer transformation runner

        # Use an internal loop for shape propagation until stabilization or max_iterations
        for iter_count in range(self.max_iterations):
            need_rerun_internal = False # Flag for this internal pass

            # Get nodes in topological order (might change between passes if nodes are removed/added)
            nodes_topo_order = list(model_current.graph.node)

            stop_reached = False
            shape_changed_in_this_pass = {} # Record shapes changed in this pass

            # Iterate through nodes to perform removal and local shape updates
            for node in nodes_topo_order:
                 if self.stop_node_name is not None and node.name == self.stop_node_name:
                     stop_reached = True
                     continue # Skip this node if it's the stop boundary

                 if stop_reached:
                     # Skip nodes after the stop boundary for pruning and initial shape updates
                     continue

                 # Process inputs and outputs of the current node
                 candidate_tensor_names = list(node.input) + list(node.output)

                 for tensor_name in candidate_tensor_names:
                      # We only prune tensors if they have a sparsity mask annotation
                      mask = model_current.get_tensor_sparsity(tensor_name)

                      if mask is None or mask == {}:
                          # If no mask, but the tensor's shape *was* changed in a previous pass,
                          # we need to see if this change affects the output of *this* node
                          # and propagate it forward. This is handled in the shape propagation block below.
                          continue # Skip sample removal if no mask

                      # --- Perform Removal if Mask Exists and is Valid ---

                      initializer = model_current.get_initializer(tensor_name)
                      io_t = initializer # Use initializer if it exists
                      io_shp = model_current.get_tensor_shape(tensor_name) # Get the tensor shape

                      current_shape = io_shp if io_t is None else (io_t.shape if hasattr(io_t, 'shape') else None)

                      if current_shape is None:
                          warnings.warn(f"Cannot prune tensor {tensor_name} with unknown shape.")
                          model_current.set_tensor_sparsity(tensor_name, {})
                          continue # Skip this tensor

                      # Process only temporal masks (axis 2)
                      temporal_mask = mask.get(2) # Get mask specifically for axis 2

                      if temporal_mask is None or not temporal_mask:
                           # No temporal mask for this tensor, clear other masks if any
                           non_temporal_masks = {ax: m for ax, m in mask.items() if ax != 2}
                           if non_temporal_masks:
                                #warnings.warn(f"Non-temporal sparsity masks found on tensor {tensor_name}. Ignoring for SamplePruning.")
                                pass # Optionally clear non-temporal masks here
                           model_current.set_tensor_sparsity(tensor_name, {})
                           continue # Skip this tensor for temporal pruning

                      # Ensure temporal_mask is a set of integers
                      temporal_mask_set = set(int(idx) for idx in temporal_mask if isinstance(idx, (int, str)) and str(idx).isdigit())

                      # Apply pruning based on the validated temporal mask
                      if io_t is None:
                          # dynamic input/output, compute new shape only
                          if 2 >= len(current_shape):
                               # Already warned if axis out of bounds for shape calc
                               model_current.set_tensor_sparsity(tensor_name, {})
                               continue # Skip

                          temp_shp = list(current_shape)
                          valid_indices_in_dim = [idx for idx in temporal_mask_set if 0 <= idx < temp_shp[2]]

                          if not valid_indices_in_dim:
                               model_current.set_tensor_sparsity(tensor_name, {})
                               continue # No valid indices to prune

                          temp_shp[2] -= len(valid_indices_in_dim)
                          new_shp = tuple(temp_shp)

                          if new_shp != current_shape:
                              model_current.set_tensor_shape(tensor_name, new_shp)
                              shape_changed_in_this_pass[tensor_name] = new_shp # Record change
                              need_rerun_internal = True # Need another pass for shape propagation
                              need_rerun_outer = True # Also signal outer transformation to rerun

                      else:
                          # initializer, remove samples from tensor data
                          if 2 >= io_t.ndim:
                              # Already warned if axis out of bounds for pruning
                              model_current.set_tensor_sparsity(tensor_name, {})
                              continue # Skip

                          valid_indices_in_dim = [idx for idx in temporal_mask_set if 0 <= idx < io_t.shape[2]]
                          if not valid_indices_in_dim:
                              model_current.set_tensor_sparsity(tensor_name, {})
                              continue # No valid indices to prune

                          new_t = remove_masked_tensor_samples(io_t, valid_indices_in_dim, axis=2)

                          if new_t.shape != io_t.shape:
                             model_current.set_initializer(tensor_name, new_t)
                             # Shape is automatically updated by set_initializer
                             shape_changed_in_this_pass[tensor_name] = new_t.shape # Record change
                             need_rerun_internal = True # Need another pass for shape propagation
                             need_rerun_outer = True # Also signal outer transformation to rerun


                      # clear sparsity annotation after processing
                      model_current.set_tensor_sparsity(tensor_name, {})


            # --- Shape Propagation within this internal pass ---
            # Now, iterate through nodes *again* in this same internal pass to propagate shapes
            # based on the changes recorded in shape_changed_in_this_pass.
            # This needs to be done after all explicit pruning is potentially done for this pass.
            # Iterate in topological order ensures shapes are propagated forward correctly.
            # We need to check if inputs *received* shapes in this pass.

            propagated_shape_changes = {} # Record shapes propagated in this block

            for node in nodes_topo_order:
                # Skip nodes that were explicitly skipped during the removal phase
                if self.stop_node_name is not None and node.name == self.stop_node_name:
                    continue
                if stop_reached and node.name != self.stop_node_name:
                    continue # Don't propagate shapes past the stop boundary either

                input_shape_updated_in_this_pass = False
                for inp_name in node.input:
                     # Check if any input's shape was changed explicitly OR propagated in this pass
                     if inp_name in shape_changed_in_this_pass or inp_name in propagated_shape_changes:
                          input_shape_updated_in_this_pass = True
                          break # Only need one changed input to trigger propagation for this node

                if input_shape_updated_in_this_pass:
                    # An input shape changed, try to re-calculate this node's output shapes
                    # based on its operation type and the *current* input shapes from the model state.

                    # For common ops affecting temporal dim (Conv, Pool):
                    if node.op_type == "Conv":
                         input_shape = model_current.get_tensor_shape(node.input[0])
                         if input_shape is None or len(input_shape) <= 2: continue

                         # Check if input_shape[2] is a known size (not symbolic/None)
                         if input_shape[2] is None or not isinstance(input_shape[2], int):
                              # Cannot calculate output shape with symbolic input temporal size
                              warnings.warn(f"Cannot propagate shape for Conv node {node.name}: input temporal size is unknown or symbolic ({input_shape[2]})")
                              continue

                         # Get Conv attributes - using default values if not found
                         stride_attr = get_by_name(node.attribute, "strides")
                         stride = stride_attr.ints[0] if stride_attr and stride_attr.ints else 1 # Temporal stride

                         dilation_attr = get_by_name(node.attribute, "dilations")
                         dilation = dilation_attr.ints[0] if dilation_attr and dilation_attr.ints else 1 # Temporal dilation

                         kernel_shape_attr = get_by_name(node.attribute, "kernel_shape")
                         kernel_size = kernel_shape_attr.ints[0] if kernel_shape_attr and kernel_shape_attr.ints else 1 # Temporal kernel size

                         padding_attr = get_by_name(node.attribute, "pads")
                         # Assuming padding is [temporal_start, spatial_start, temporal_end, spatial_end]
                         padding_start = padding_attr.ints[0] if padding_attr and padding_attr.ints and len(padding_attr.ints) > 0 else 0
                         padding_end = padding_attr.ints[2] if padding_attr and padding_attr.ints and len(padding_attr.ints) > 2 else padding_start


                         # Calculate output temporal size
                         input_temporal_size = input_shape[2]
                         output_temporal_size = int(np.floor((input_temporal_size + padding_start + padding_end - dilation * (kernel_size - 1) - 1) / stride + 1))

                         # Update the output tensor shapes if they change
                         for out_name in node.output:
                              output_shape_current = list(model_current.get_tensor_shape(out_name))
                              if len(output_shape_current) > 2: # Ensure output has a temporal dimension
                                  new_output_shape = list(output_shape_current)
                                  new_output_shape[2] = output_temporal_size
                                  new_output_shape_tuple = tuple(new_output_shape)

                                  if new_output_shape_tuple != tuple(output_shape_current):
                                       model_current.set_tensor_shape(out_name, new_output_shape_tuple)
                                       propagated_shape_changes[out_name] = new_output_shape_tuple # Record propagated change
                                       need_rerun_internal = True # Need another pass

                    elif node.op_type in ["MaxPool", "AveragePool"]:
                         input_shape = model_current.get_tensor_shape(node.input[0])
                         if input_shape is None or len(input_shape) <= 2: continue

                         if input_shape[2] is None or not isinstance(input_shape[2], int):
                             warnings.warn(f"Cannot propagate shape for Pool node {node.name}: input temporal size is unknown or symbolic ({input_shape[2]})")
                             continue


                         # Get Pool attributes
                         kernel_shape_attr = get_by_name(node.attribute, "kernel_shape")
                         kernel_size = kernel_shape_attr.ints[0] if kernel_shape_attr and kernel_shape_attr.ints else 1 # Temporal kernel size

                         stride_attr = get_by_name(node.attribute, "strides")
                         stride = stride_attr.ints[0] if stride_attr and stride_attr.ints else 1 # Temporal stride

                         padding_attr = get_by_name(node.attribute, "pads")
                         padding_start = padding_attr.ints[0] if padding_attr and padding_attr.ints and len(padding_attr.ints) > 0 else 0
                         padding_end = padding_attr.ints[2] if padding_attr and padding_attr.ints and len(padding_attr.ints) > 2 else padding_start

                         # Calculate output temporal size
                         input_temporal_size = input_shape[2]
                         output_temporal_size = int(np.floor((input_temporal_size + padding_start + padding_end - kernel_size) / stride + 1))

                         # Update the output tensor shape
                         for out_name in node.output:
                              output_shape_current = list(model_current.get_tensor_shape(out_name))
                              if len(output_shape_current) > 2: # Ensure output has a temporal dimension
                                  new_output_shape = list(output_shape_current)
                                  new_output_shape[2] = output_temporal_size
                                  new_output_shape_tuple = tuple(new_output_shape)

                                  if new_output_shape_tuple != tuple(output_shape_current):
                                       model_current.set_tensor_shape(out_name, new_output_shape_tuple)
                                       propagated_shape_changes[out_name] = new_output_shape_tuple # Record
                                       need_rerun_internal = True # Need another pass

                    # Add more op types here that affect temporal dimensions
                    # For Elementwise operations (Add, Mul, etc.), shapes should be the same as inputs (broadcasting handled implicitly)
                    # For Concat along axis 2, new shape is sum of input shapes along axis 2
                    # For other ops that don't change temporal dimension (e.g., BatchNormalization, ReLU, Quant, Dropout),
                    # the output shape temporal dimension should be the same as the input temporal dimension.
                    # If input shape[2] changed, and the op doesn't change temporal size, output shape[2] = input_shape[2]
                    elif node.op_type in ["Add", "Mul", "BatchNormalization", "Relu", "Quant", "QuantizeLinear", "DequantizeLinear", "Clip", "Dropout"]:
                        input_shape = model_current.get_tensor_shape(node.input[0])
                        if input_shape is None or len(input_shape) <= 2 or input_shape[2] is None or not isinstance(input_shape[2], int):
                             # Cannot propagate if input temporal size is unknown or symbolic
                             continue

                        input_temporal_size = input_shape[2]

                        for out_name in node.output:
                           output_shape_current = list(model_current.get_tensor_shape(out_name))
                           if len(output_shape_current) > 2:
                                new_output_shape = list(output_shape_current)
                                new_output_shape[2] = input_temporal_size
                                new_output_shape_tuple = tuple(new_output_shape)

                                if new_output_shape_tuple != tuple(output_shape_current):
                                     model_current.set_tensor_shape(out_name, new_output_shape_tuple)
                                     propagated_shape_changes[out_name] = new_output_shape_tuple
                                     need_rerun_internal = True

                    # Add more specific op handling here if needed...
                    # Default: For ops not explicitly handled, assume output temporal dimension is same as first input
                    # else:
                    #      input_shape = model_current.get_tensor_shape(node.input[0])
                    #      if input_shape is None or len(input_shape) <= 2 or input_shape[2] is None or not isinstance(input_shape[2], int):
                    #           continue
                    #      input_temporal_size = input_shape[2]
                    #      for out_name in node.output:
                    #           output_shape_current = list(model_current.get_tensor_shape(out_name))
                    #           if len(output_shape_current) > 2:
                    #                new_output_shape = list(output_shape_current)
                    #                new_output_shape[2] = input_temporal_size
                    #                new_output_shape_tuple = tuple(new_output_shape)
                    #                if new_output_shape_tuple != tuple(output_shape_current):
                    #                     model_current.set_tensor_shape(out_name, new_output_shape_tuple)
                    #                     propagated_shape_changes[out_name] = new_output_shape_tuple
                    #                     need_rerun_internal = True


            # After iterating through all nodes for shape propagation, update the list of changed shapes for the next internal pass
            shape_changed_in_this_pass = propagated_shape_changes # shapes propagated become sources for next pass


            if not need_rerun_internal:
                 # No shapes changed in this pass, stabilization reached
                 print(f"Shape propagation stabilized after {iter_count + 1} internal passes.")
                 break # Exit the internal while loop
            else:
                 # If max_iterations reached and still changing, warn.
                 if iter_count == self.max_iterations - 1:
                     warnings.warn(f"Max shape propagation iterations ({self.max_iterations}) reached before stabilization.")

        # After the internal shape propagation loop finishes (stabilized or maxed out)
        # The outer transformation runner might still call apply again if need_rerun_outer was set.
        return (model_current, need_rerun_outer)


class PruneSamples(Transformation):
    """Top-level transformation to prune temporal samples with stop node functionality."""

    def __init__(self, prune_spec: Dict, lossy: bool = True, stop_node_name=None, max_shape_propagation_iterations=100) -> None:
        super().__init__()
        self.prune_spec = prune_spec
        self.lossy = lossy
        self.stop_node_name = stop_node_name
        self.max_shape_propagation_iterations = max_shape_propagation_iterations

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:

        # Apply initial masks (these are on the specified tensors, likely outputs)
        # Ensure initial masks have integer keys and lists are converted to sets
        prune_spec_processed = {}
        for tensor_name, axis_map in self.prune_spec.items():
             prune_spec_processed[tensor_name] = {}
             for axis_str, indices_list in axis_map.items():
                 try:
                     axis_int = int(axis_str)
                     prune_spec_processed[tensor_name][axis_int] = set(indices_list) # Ensure indices are a set
                 except ValueError:
                      warnings.warn(f"Sparsity mask key '{axis_str}' for tensor {tensor_name} in prune_spec is not an integer. Ignoring.")


        model = model.transform(ApplyTemporalMasks(prune_spec_processed))

        # Removed PropagateTemporalMasks transformation

        # Remove masked samples and propagate shape changes
        # This transformation will now iterate until all shapes are stable after pruning
        # Pass the max_iterations safeguard to RemoveMaskedSamples
        model = model.transform(RemoveMaskedSamples(self.lossy, stop_node_name=self.stop_node_name, max_iterations=self.max_shape_propagation_iterations))

        # Return True if the removal/propagation step needed rerunning (though the internal loop might have handled most of it)
        # For simplicity and to match the common QONNX pattern, we can return False here
        # if we are confident RemoveMaskedSamples handles internal stabilization.
        # However, returning the inner need_rerun is technically more correct for the outer runner.
        # Let's return the inner need_rerun for robustness, although it might mean multiple calls to PruneSamples.apply
        # if RemoveMaskedSamples didn't fully stabilize.
        # Let's stick to returning False at the top level as per the original PruneChannels, assuming
        # RemoveMaskedSamples internal loop and the final InferShapes handle stabilization.

        # A simpler approach for the outer PruneSamples transformation: just call the steps.
        # The rerunning logic is now *inside* RemoveMaskedSamples.
        # So PruneSamples itself doesn't need to return need_rerun=True for the outer loop.
        return (model, False) # PruneSamples itself doesn't need rerunning