# -*- coding: utf-8 -*-

import json
import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

# --- Assume TemporalBlock2d and UnsignedQuantIdentity from original code are available ---
# Or define them here if not imported:
class UnsignedQuantIdentity(qnn.QuantIdentity):
    def forward(self, x):
        qx = super().forward(x)
        clamped_val = torch.clamp(qx.value, 0, 127)
        return qx._replace(value=clamped_val)

class TemporalBlock2d(nn.Module):
    # ... (Define __init__ and forward exactly as in your original code) ...
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, stride, dropout=0.05):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride

        self.conv1 = qnn.QuantConv2d(
            n_inputs, n_outputs,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(0, 0),
            dilation=(dilation, 1),
            weight_quant=Int8WeightPerTensorFloat,
            input_quant=Int8ActPerTensorFloat,
            weight_bit_width=8,
            act_bit_width=8,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = qnn.QuantConv2d(
            n_outputs, n_outputs,
            kernel_size=(kernel_size, 1),
            stride=(1, 1), # NOTE: This stride is 1 in TCN paper for residual connection
            padding=(0, 0),
            dilation=(dilation, 1), # Dilation is block-level
            weight_quant=Int8WeightPerTensorFloat,
            input_quant=Int8ActPerTensorFloat,
            weight_bit_width=8,
            act_bit_width=8,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        self.relu_after_conv = qnn.QuantReLU(return_quant_tensor=True)

    def forward(self, x):
        # Order from your original code seems to be Conv1 -> BN1 -> Dropout1,
        # then Conv2 -> BN2 -> ReLU -> Dropout2.
        # TCN blocks often use residual connection *after* the full 2-conv sequence.
        # Ensure this matches YOUR training model precisely if not using residual.
        x = self.conv1(x)
        x = self.bn1(x)
        # Assuming dropout *after* activation as is more common
        # x = self.dropout1(x) # Keep or move this based on training model

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu_after_conv(x).value # QuantReLU and get value
        # x = self.dropout2(x) # Keep or move this based on training model

        # Apply dropout based on where it was in training
        # Example assuming Dropout after first BN, then after ReLU:
        # out = self.dropout1(self.bn1(self.conv1(x_in)))
        # out = self.dropout2(self.relu_after_conv(self.bn2(self.conv2(out)))).value
        # THIS Clean block doesn't do residual, just sequence matching YOUR provided code
        # Let's stick to the simple sequence as per your provided `TemporalBlock2d`
        # If trained order was: Conv1 -> BN1 -> Dropout1, then Conv2 -> BN2 -> ReLU -> Dropout2
        # Then the clean block forward IS:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu_after_conv(x).value
        x = self.dropout2(x)
        return x

class TCN2d(nn.Module):
    def __init__(self, custom_blocks: list, num_outputs: int):
        super(TCN2d, self).__init__()
        self.temporal_blocks = nn.ModuleList(custom_blocks)
        last_out_channels = custom_blocks[-1].conv2.out_channels

        # We also need a 1x1 conv to get to num_outputs
        self.fc = qnn.QuantConv2d(
            in_channels=last_out_channels,
            out_channels=num_outputs,
            kernel_size=(1, 1),
            weight_quant=Int8WeightPerTensorFloat,
            input_quant=Int8ActPerTensorFloat,
            weight_bit_width=8,
            act_bit_width=8,
            bias=False
        )
        

        # Input quant layer
        self.inp_quant = qnn.QuantIdentity(
            bit_width=8,
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        self.output_quant = UnsignedQuantIdentity(
            bit_width=8,
            act_quant=Int8ActPerTensorFloat,  # still using Int8 under the hood
            return_quant_tensor=True
        )

# Inside TCN2d_Training_Architecture.forward

    def forward(self, x):
        # print("--- TCN2d_Training_Architecture Forward Start ---")
        # print(f"Initial Input shape: {x.shape}, dtype: {x.dtype}")

        x = self.inp_quant(x) # Output is QuantTensor
        # print(f"After inp_quant: type: {type(x)}, shape: {x.shape}")
        x = x.value # Get the value tensor
        # print(f"After inp_quant.value: type: {type(x)}, shape: {x.shape}")


        # Process temporal blocks
        # print("\n--- Entering Temporal Blocks Loop ---")
        for i, block in enumerate(self.temporal_blocks):
            # Print input shape right before calling the block
            print(f"\n  Before temporal_blocks[{i}] call: Input shape {x.shape}, dtype: {x.dtype}")

            # --- Add prints inside TemporalBlock2d_inf_clean.forward ---
            # This is where the error happens for i = 0
            # Temporarily modify TemporalBlock2d_inf_clean's forward:
            #
            #  def forward(self, x):
            #     print(f"\n    Inside TemporalBlock2d_inf_clean (Block {i if 'i' in locals() else '?'}) before conv1: Input shape {x.shape}, dtype: {x.dtype}")
            #     x = self.conv1(x) # ERROR OCCURS HERE FOR THE FIRST BLOCK (i=0)
            #     print(f"    Inside TemporalBlock2d_inf_clean (Block {i if 'i' in locals() else '?'}) after conv1: Output shape {x.shape}, dtype: {x.dtype}")
            #     # ... rest of block forward ...


            x = block(x) # This is where the error is reported
            # print(f"  After temporal_blocks[{i}] call: Output shape {x.shape}, dtype: {x.dtype}")

        # print("\n--- After Temporal Blocks Loop ---")
        # ... rest of forward ...

# Run the script again with these prints.
         
        #print("Shape before slicing:", x.shape)

        # Select the center index along time dim (dim=2), assuming x.shape == [N, C, T, 1]
        center_idx = 84
        x = x[:, :, center_idx:center_idx+1, :]  # keeps dimensions
        # index = torch.tensor([center_idx], dtype=torch.long, device=x.device)
        # x = torch.index_select(x, dim=2, index=index)
        x = self.fc(x)
        x = self.output_quant(x).value
        x = x.reshape(x.size(0), -1)


        return x
# Use the clean version derived before, matching your sequence without internal slicers
class TemporalBlock2d_inf_clean(nn.Module):
     def __init__(self, n_inputs, n_outputs, kernel_size, dilation, stride, dropout=0.2):
        super().__init__()
        self.kernel_size = kernel_size # Store these for dynamic change check
        self.dilation = dilation
        self.stride = stride

        self.conv1 = qnn.QuantConv2d(n_inputs, n_outputs, (kernel_size, 1), stride=(stride, 1), padding=(0, 0), dilation=(dilation, 1), weight_quant=Int8WeightPerTensorFloat, input_quant=Int8ActPerTensorFloat, weight_bit_width=8, act_bit_width=8, bias=False)
        self.bn1 = nn.BatchNorm2d(n_outputs)
        self.dropout1 = nn.Dropout(dropout) # Dropout after BN as in your forward

        self.conv2 = qnn.QuantConv2d(n_outputs, n_outputs, (kernel_size, 1), stride=(1, 1), padding=(0, 0), dilation=(dilation, 1), weight_quant=Int8WeightPerTensorFloat, input_quant=Int8ActPerTensorFloat, weight_bit_width=8, act_bit_width=8, bias=False)
        self.bn2 = nn.BatchNorm2d(n_outputs)
        self.dropout2 = nn.Dropout(dropout) # Dropout after ReLU as in your forward
        self.relu_after_conv = qnn.QuantReLU(return_quant_tensor=True)

     def forward(self, x):
        # Order as provided in your TCN example TemporalBlock2d
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout1(x) # Dropout after first BN

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu_after_conv(x).value # ReLU after second BN
        x = self.dropout2(x) # Dropout after ReLU
        return x


# --- Direct Slicer (Helper Module) ---
class DirectSlice(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, indices_to_select):
        """Applies slicing directly using provided indices (must be LongTensor)."""
        if indices_to_select is None or indices_to_select.numel() == 0:
            # If indices are empty, maybe return an empty tensor of appropriate shape?
            # Or pass through? If intended for selection, empty usually means prune all.
            # Let's return empty as that reflects the intended pruning.
            # An empty slice like x[:, :, [] , :] gives shape (N, C, 0, H)
            # print(f"DirectSlice: Indices empty or None, returning empty slice.")
            return x[:, :, [], :] # Explicitly slice with empty list

        # Ensure indices are LongTensor and on the correct device
        if not isinstance(indices_to_select, torch.Tensor):
             indices_to_select = torch.tensor(indices_to_select, dtype=torch.long, device=x.device)
        else: # Already a tensor
             if indices_to_select.dtype != torch.long:
                  indices_to_select = indices_to_select.to(torch.long)
             indices_to_select = indices_to_select.to(x.device)

        # Basic bounds check before slicing
        current_len = x.shape[2]
        if indices_to_select.numel() > 0:
            max_idx = torch.max(indices_to_select)
            if max_idx >= current_len:
                 print(f"\n‼️ ERROR in DirectSlice: Max index {max_idx.item()} is out of bounds for tensor dim size {current_len}.")
                 # Filter out invalid indices and print warning, or raise error.
                 # For numerical comparison, better to raise to flag mismatch immediately.
                 raise IndexError(f"Max index {max_idx.item()} out of bounds for dim {current_len}")
            # Min index check
            min_idx = torch.min(indices_to_select)
            if min_idx < 0:
                 print(f"\n‼️ ERROR in DirectSlice: Min index {min_idx.item()} is negative.")
                 raise IndexError(f"Min index {min_idx.item()} is negative")


        try:
             # print(f"DirectSlice: Applying {indices_to_select.numel()} indices to shape {x.shape}")
             result = torch.index_select(x, dim=2, index=indices_to_select) # Use index_select for list of indices
             # print(f"DirectSlice: Output shape {result.shape}")
             return result

        except IndexError as e:
            print(f"\n‼️ ERROR applying indices in DirectSlice. Shape={x.shape}, Num Indices: {indices_to_select.numel()}, Indices (first 10): {indices_to_select[:min(10, indices_to_select.numel())]}, Max Index: {torch.max(indices_to_select).item() if indices_to_select.numel()>0 else 'N/A'}, Min Index: {torch.min(indices_to_select).item() if indices_to_select.numel()>0 else 'N/A'}")
            raise e


# --- Inference TCN with Remapping & Dynamic Dilation ---
class TCN2d_inf_iterative_pruning(nn.Module):
    def __init__(self, custom_blocks_config: list, num_outputs: int, corrected_abs_map_json_path: str):
        super().__init__()
        # Load the corrected_relevant_indices_per_model_layer.json
        self.pruning_map = {} # Stores ABSOLUTE original indices needed from layer outputs
        try:
            with open(corrected_abs_map_json_path, "r") as f:
                pruning_data = json.load(f)
            # Convert lists to sorted torch.LongTensor and move to CPU initially
            for key, value in pruning_data.items():
                 if isinstance(value, list):
                      self.pruning_map[key] = torch.tensor(sorted(list(set(value))), dtype=torch.long)
                 else:
                     self.pruning_map[key] = None # Indicate invalid entry
            # print(f"Loaded and parsed map from {corrected_abs_map_json_path}")
            # print(f"Map keys: {self.pruning_map.keys()}")
            # Example indices check
            # if 'global_in' in self.pruning_map:
            #     print(f"global_in indices: {self.pruning_map['global_in'].numel()}")
            # if 'temporal_blocks.0.conv2' in self.pruning_map:
            #      print(f"temporal_blocks.0.conv2 indices: {self.pruning_map['temporal_blocks.0.conv2'].numel()}")


        except FileNotFoundError:
             print(f"‼️ ERROR: Map file not found at {corrected_abs_map_json_path}. Disabling slicing.")
             self.pruning_map = {} # Disable slicing if map is missing
        except json.JSONDecodeError:
             print(f"‼️ ERROR: Map file at {corrected_abs_map_json_path} is not valid JSON. Disabling slicing.")
             self.pruning_map = {} # Disable slicing

        # Store original block parameters accessible by layer name
        self.original_layer_configs = {}
        self.temporal_blocks = nn.ModuleList()
        # Assuming blocks_config lists args in order block0, block1, block2...
        layer_names_template = [
             f'temporal_blocks.{i}.conv1' for i in range(len(custom_blocks_config))
        ] + [
             f'temporal_blocks.{i}.conv2' for i in range(len(custom_blocks_config))
        ]
        layer_names_template = sorted(list(set(layer_names_template))) # Ensure unique names if template overlaps

        for i, block_args in enumerate(custom_blocks_config):
            # Use the clean block version
            block = TemporalBlock2d_inf_clean(**block_args)
            self.temporal_blocks.append(block)
            # Store original dilation/stride/kernel for later lookup and dynamic change
            # Assuming layers within the block are named 'conv1', 'conv2' matching map keys
            # Ensure key names match EXACTLY those in your corrected_relevant_indices_per_model_layer.json
            self.original_layer_configs[f'temporal_blocks.{i}.conv1'] = {
                'dilation': block.conv1.dilation[0], 'stride': block.conv1.stride[0], 'kernel_size': block.kernel_size # kernel_size is block-level property
            }
             # conv2 always stride 1 in this setup
            self.original_layer_configs[f'temporal_blocks.{i}.conv2'] = {
                'dilation': block.conv2.dilation[0], 'stride': block.conv2.stride[0], 'kernel_size': block.kernel_size
            }


        last_out_channels = custom_blocks_config[-1]['n_outputs']

        # FC Layer - ensure in_channels matches last block output channels.
        # Its kernel_size MUST be (1,1) if the preceding slice results in T=1.
        self.fc = qnn.QuantConv2d(
            in_channels=last_out_channels, out_channels=num_outputs, kernel_size=(1, 1),
            weight_quant=Int8WeightPerTensorFloat, input_quant=Int8ActPerTensorFloat,
            weight_bit_width=8, act_bit_width=8, bias=False
        )

        # Input/Output Quantizers
        self.inp_quant = qnn.QuantIdentity(bit_width=8, act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.output_quant = UnsignedQuantIdentity(bit_width=8, act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)

        # --- Create Slicer Modules ---
        # Map keys from JSON to slicers. We slice *after* the tensor corresponding to the key is produced.
        # Need a slicer for global_in (before 1st block), and after each conv output.
        self.input_slicer = DirectSlice()
        # Slicers after conv1 and conv2 in each block
        self.slicers = nn.ModuleDict({})
        for i in range(len(self.temporal_blocks)):
            self.slicers[f'temporal_blocks_{i}_conv1'] = DirectSlice() # Slicer after conv1 output
            self.slicers[f'temporal_blocks_{i}_conv2'] = DirectSlice() # Slicer after conv2 output (block output)


    def forward(self, x):
        # --- Load & Move Pruning Indices to Device ---
        # Load happens in __init__, move to device here in forward
        # Check if pruning map was loaded successfully
        if not self.pruning_map:
             # print("Slicing disabled, running original network path (less efficient)")
             # If slicing disabled, potentially run original logic or handle as error.
             # For comparison, let's just print a warning and proceed without slicing.
             pass # Need to implement the non-slicing path if needed, for now assume map is present


        # At each stage, we need the indices *needed* from the tensor at this point.
        # These needed indices are looked up using keys like 'global_in', 'temporal_blocks.0.conv1', etc.

        # --- Stage 0: Input ---
        # Indices needed from the *original input* for block0.conv1 (key 'global_in')
        original_indices_needed_input = self.pruning_map.get("global_in", None)

        if original_indices_needed_input is not None and original_indices_needed_input.numel() > 0:
            # print(f"Applying Input Slice ('global_in'), original_indices_needed: {original_indices_needed_input.numel()}")
            # For the input, the "remapped" indices are just the original indices relative to index 0.
            indices_to_select = original_indices_needed_input.to(x.device)
            x = self.input_slicer(x, indices_to_select)
            # print(f"After Input Slice: Shape={x.shape}")
        # else: print("No input slice specified or indices empty.")


        # Input Quant
        # QuantIdentity takes QuantTensor or Tensor, should be fine with sliced Tensor
        x_quant = self.inp_quant(x) # Output is QuantTensor

        # Important: Extract value only *after* passing through the QuantIdentity layer
        x = x_quant.value


        # --- Process Temporal Blocks and Intermediate Slicing ---

        for block_idx in range(len(self.temporal_blocks)):
             block = self.temporal_blocks[block_idx]

             # === Conv1 -> BN -> Dropout ===
             layer_key_conv1 = f'temporal_blocks.{block_idx}.conv1'
             conv1_module = block.conv1
             original_config_conv1 = self.original_layer_configs.get(layer_key_conv1) # Use get for safety

             # Store original dilation tuple
             if original_config_conv1: # Check if config exists
                 original_dilation_conv1 = original_config_conv1['dilation']
                 original_stride_conv1 = original_config_conv1['stride']
                 original_conv1_dilation_tuple = conv1_module.dilation # Store actual tuple

                 # ** Set Dilation to 1 if applicable **
                 # Apply this only if original D > 1 AND original S == 1.
                 # block0.conv1 has S=2, block1.conv1 has S=1, block2.conv1 has S=1
                 # So potential candidates: temporal_blocks.1.conv1, temporal_blocks.2.conv1
                 if original_dilation_conv1 > 1 and original_stride_conv1 == 1:
                      # print(f"DEBUG: Setting dilation for {layer_key_conv1} from {original_dilation_conv1} to 1")
                      conv1_module.dilation = (1, 1)


             x = conv1_module(x) # Pass current x to conv1


             # ** Restore Dilation immediately after conv1 forward **
             if original_config_conv1: # Check if config exists
                 if original_dilation_conv1 > 1 and original_stride_conv1 == 1:
                       conv1_module.dilation = original_conv1_dilation_tuple # Restore

             # Apply subsequent layers in block1.conv1's pipeline: BN1, Dropout1
             x = block.bn1(x)
             x = block.dropout1(x) # Dropout after BN

             # --- Slice AFTER Conv1 output ---
             # Look up needed indices from the original output of this layer (conv1)
             slice_key_conv1_out = layer_key_conv1 # Key in pruning_map is the layer name whose output we slice
             original_indices_needed_from_conv1_out = self.pruning_map.get(slice_key_conv1_out, None)

             if original_indices_needed_from_conv1_out is not None and original_indices_needed_from_conv1_out.numel() > 0:
                 # !!! This is the step where remapping is needed !!!
                 # The indices `original_indices_needed_from_conv1_out` are ABSOLUTE relative to
                 # the ORIGINAL output of conv1 (layer L1, L3, or L5).
                 # The tensor `x` here is the OUTPUT of conv1, but potentially sliced earlier.
                 # The indices to select FROM `x` must be relative to `x`.

                 # We need a function to translate: "original index J in layer L output" -> "index K in the current tensor x"
                 # This function IS the forward pass simulation logic missing before!

                 # Let's *use* the indices from the "remapped" map file instead.
                 # That remapping script calculated these relative indices already.
                 # Adapt __init__ to load remapped map. Let's pass that file path instead.
                 # OR, change __init__ to *generate* the remapped map using the loaded abs map.

                 # Let's pass the remapped map JSON path instead in __init__
                 # Redefine __init__ again!

                 # Okay, the original approach of passing the corrected_abs_map_json_path to __init__
                 # is fine, but __init__ needs to *internally* generate the remapped_map
                 # and store THAT for forward pass slicing.

                 # Need `create_remapped_node_index_map` helper *inside* or *called by* __init__
                 # And the helper `remap_indices` needs to be part of the class or scope.

                 # Redefining __init__ and adding remap/generate logic... this is iterative refinement!

                 return self._forward_with_iterative_pruning(x) # Jump to refactored forward


    # --- New forward pass implementing the logic with remapped indices ---
    def _forward_with_iterative_pruning(self, x):
        # Ensure remapped map is generated/loaded
        if not hasattr(self, '_is_remapped_map_ready'):
             self._prepare_remapped_map() # Method to load/generate remapped map

        if not self.remapped_pruning_map:
            # Fallback or error if map isn't ready
            print("‼️ ERROR: Remapped pruning map not ready. Cannot run iterative pruning.")
            # Maybe raise error or return full model output? For numerical check, maybe crash is better.
            # Fallback: run original (non-pruned) path conceptually - remove slices & dilation changes
            # For this exercise, let's assume map is critical. Raise error or assert.
            assert self.remapped_pruning_map, "Remapped pruning map failed to load/generate."


        # --- Stage 0: Input ---
        slice_key_input = "global_in"
        # The map now holds indices *relative to the tensor at this stage*.
        indices_to_select_input = self.remapped_pruning_map.get(slice_key_input, None)

        if indices_to_select_input is not None and indices_to_select_input.numel() > 0:
             x = self.input_slicer(x, indices_to_select_input.to(x.device))
             # print(f"After Input Slice ('global_in'): Shape={x.shape}")


        # Input Quant
        x = self.inp_quant(x).value # Extract value

        # --- Process Temporal Blocks and Intermediate Slicing ---
        slice_points_execution_order = [ # Keys from remapped_pruning_map applied AFTER the corresponding layer output
             'temporal_blocks.0.conv1',
             'temporal_blocks.0.conv2', # Block 0 output
             'temporal_blocks.1.conv1',
             'temporal_blocks.1.conv2', # Block 1 output
             'temporal_blocks.2.conv1',
             'temporal_blocks.2.conv2', # Block 2 output (final selection)
        ]

        current_slice_point_idx = 0 # Index into slice_points_execution_order

        for block_idx in range(len(self.temporal_blocks)):
             block = self.temporal_blocks[block_idx] # TemporalBlock2d_inf_clean instance


             # --- Process Layers Inside Block ---
             layers_in_block = [(block.conv1, block.bn1, block.dropout1), (block.conv2, block.bn2, block.relu_after_conv, block.dropout2)]
             layer_keys = [f'temporal_blocks.{block_idx}.conv1', f'temporal_blocks.{block_idx}.conv2']

             for layer_in_block_idx, (conv_module, *subsequent_modules) in enumerate(layers_in_block):

                 layer_key = layer_keys[layer_in_block_idx]
                 original_config = self.original_layer_configs.get(layer_key)

                 # === Conv Operation ===
                 # Store original dilation tuple before potential change
                 original_dilation_tuple = conv_module.dilation
                 original_dilation = original_config['dilation']
                 original_stride = original_config['stride']

                 # ** Change Dilation to 1 if applicable **
                 # Apply if original D > 1 AND original S == 1.
                 # Candidate layers: temporal_blocks.1.conv1, .conv2, temporal_blocks.2.conv1, .conv2
                 # (block_idx >= 1 for conv1, block_idx >= 1 for conv2 as block0.conv2 D=1, S=1 anyway)
                 # Refined condition based on your architecture:
                 # temporal_blocks.0.conv1: D=1, S=2 -> No change
                 # temporal_blocks.0.conv2: D=1, S=1 -> No change (D!=1 initially, condition D > 1 fails)
                 # temporal_blocks.1.conv1: D=4, S=1 -> Change (D>1 and S=1)
                 # temporal_blocks.1.conv2: D=4, S=1 -> Change (D>1 and S=1)
                 # temporal_blocks.2.conv1: D=16, S=1 -> Change (D>1 and S=1)
                 # temporal_blocks.2.conv2: D=16, S=1 -> Change (D>1 and S=1)

                 should_change_dilation = (original_dilation > 1 and original_stride == 1)

                 if should_change_dilation:
                     # print(f"DEBUG: Setting dilation for {layer_key} from {original_dilation} to 1")
                     conv_module.dilation = (1, 1)


                 x = conv_module(x) # Pass current x to the conv layer

                 # ** Restore Dilation immediately after conv forward **
                 if should_change_dilation:
                     conv_module.dilation = original_dilation_tuple # Restore

                 # === Apply Subsequent Layers in Pipeline (BN, ReLU, Dropout) ===
                 # This order must match the *training* block forward pass exactly.
                 # Based on your `TemporalBlock2d` forward:
                 # Conv1 -> BN1 -> Dropout1
                 # Conv2 -> BN2 -> ReLU -> Dropout2
                 if layer_in_block_idx == 0: # conv1 pipeline
                     x = block.bn1(x)
                     x = block.dropout1(x) # Dropout after BN
                 elif layer_in_block_idx == 1: # conv2 pipeline
                      x = block.bn2(x)
                      x = block.relu_after_conv(x).value # ReLU after BN
                      x = block.dropout2(x) # Dropout after ReLU


                 # --- Apply Slice AFTER This Layer's Output ---
                 # The key for slicing is the layer name whose output was just computed.
                 slice_key_after_layer = layer_key # This layer's name
                 indices_to_select = self.remapped_pruning_map.get(slice_key_after_layer, None)

                 if indices_to_select is not None and indices_to_select.numel() > 0:
                     # print(f"Applying Slice ('{slice_key_after_layer}') with {indices_to_select.numel()} indices to shape {x.shape}")
                     x = self.slicers[slice_key_after_layer](x, indices_to_select.to(x.device))
                     # print(f"  -> New shape: {x.shape}")
                 # else: print(f"No slice specified for {slice_key_after_layer} output.")


        # --- Final FC Layer ---
        # The last slice point was temporal_blocks.2.conv2
        # The tensor 'x' should now have length 1 along dimension 2 if the last slice
        # in the remapped map for 'temporal_blocks.2.conv2' selected only 1 index (e.g. [0]).
        # Check the final shape consistency before FC
        if x.shape[2] != 1:
             print(f"\n‼️ Warning: Final tensor shape before FC is {x.shape}. FC expects T=1.")
             # This could indicate an error in the corrected/remapped index calculation
             # for the last layer, or the definition of FC.

        x = self.fc(x) # Pass current x to the FC layer

        # Output Quant
        x = self.output_quant(x).value

        # Reshape
        x = x.reshape(x.size(0), -1) # Flatten to [N, num_outputs]

        return x

    # --- Helper method to load and generate remapped map ---
    # Call this from __init__ to prepare the map used by forward
    def _prepare_remapped_map(self):
         # Ensure this only runs once
         if hasattr(self, '_is_remapped_map_ready') and self._is_remapped_map_ready:
              return

         # Needs the corrected absolute map already loaded in __init__ (self.pruning_map)
         if not self.pruning_map:
             print("Cannot prepare remapped map: Absolute map not loaded.")
             return

         # --- Function to get Absolute Input Indices needed for Absolute Output Indices ---
         # This is the core reverse mapping logic missing earlier.
         # input_layer_name: Key like 'temporal_blocks.X.convY'
         # output_indices_abs: Absolute indices required in the output of this layer
         def get_abs_input_influence_recursive(input_layer_name, output_indices_abs, layers_config_map):
             if not output_indices_abs or output_indices_abs.numel() == 0:
                 return torch.tensor([], dtype=torch.long)

             config = layers_config_map.get(input_layer_name)
             if not config:
                  print(f"Warning: Config not found for layer {input_layer_name}. Cannot trace back.")
                  return torch.tensor([], dtype=torch.long)

             kernel_size = config['kernel_size']
             dilation = config['dilation']
             stride = config['stride']

             all_influenced_input_indices = []
             for output_idx_abs in output_indices_abs.tolist():
                 # For each required absolute output index, find the absolute input indices
                 # that produced it. Formula: input_start_index = output_idx_abs * stride.
                 # Then consider kernel * dilation spread.
                 input_start_from_output = output_idx_abs * stride
                 influenced_indices = [input_start_from_output + k * dilation for k in range(kernel_size)]
                 all_influenced_input_indices.extend(influenced_indices)

             # Need to intersect these influenced indices with the *valid range* of this layer's input.
             # This requires knowing the original length of the layer's input tensor.
             # This means simulating forward passes to get lengths. Let's rely on pre-calculated lengths from indices.py.

             # Original lengths: original_input_length -> L1 -> L2 -> L3 -> L4 -> L5 -> L6
             original_lengths_map = {} # Map layer key -> original input length
             # This needs careful definition based on your architecture flow.
             # Original input length:
             # temporal_blocks.0.conv1 takes original_input
             # temporal_blocks.0.conv2 takes output of temporal_blocks.0.conv1 (L1)
             # temporal_blocks.1.conv1 takes output of temporal_blocks.0.conv2 (L2)
             # ...and so on.

             # To avoid recalculating lengths, let's load them or pass them.
             # Assume you save output lengths from indices_corrected.py run into a JSON.
             try:
                 with open("calculated_original_lengths.json", "r") as f:
                      orig_lens = json.load(f)
                 original_lengths_map_loaded = {
                     'global_input': orig_lens.get('original_input_length'),
                     'temporal_blocks.0.conv1': orig_lens.get('output_length_L1'), # Input to block0.conv2 is L1 output
                     'temporal_blocks.0.conv2': orig_lens.get('output_length_L2'), # Input to block1.conv1 is L2 output
                     'temporal_blocks.1.conv1': orig_lens.get('output_length_L3'), # Input to block1.conv2 is L3 output
                     'temporal_blocks.1.conv2': orig_lens.get('output_length_L4'), # Input to block2.conv1 is L4 output
                     'temporal_blocks.2.conv1': orig_lens.get('output_length_L5'), # Input to block2.conv2 is L5 output
                     # The last conv output length (L6) is needed too, but not as input len here
                     'temporal_blocks.2.conv2_output': orig_lens.get('output_length_L6')
                 }
             except FileNotFoundError:
                  print("‼️ ERROR: calculated_original_lengths.json not found. Cannot map inputs properly.")
                  return torch.tensor([], dtype=torch.long) # Fatal for map generation

             input_length_for_this_layer = original_lengths_map_loaded.get(input_layer_name, None)
             if input_length_for_this_layer is None:
                  # This layer is the very first (temporal_blocks.0.conv1), its input is global input
                  if input_layer_name == 'temporal_blocks.0.conv1':
                      input_length_for_this_layer = original_lengths_map_loaded.get('global_input')
                  else:
                       print(f"‼️ Warning: Original input length for layer {input_layer_name} not found.")
                       return torch.tensor([], dtype=torch.long) # Cannot proceed


             # Filter indices to be within the valid input range for this original layer
             valid_input_indices = [idx for idx in all_influenced_input_indices if 0 <= idx < input_length_for_this_layer]

             return torch.tensor(sorted(list(set(valid_input_indices))), dtype=torch.long)


         # --- GENERATE the Remapped Map by Simulating Forward with Requisite Inputs ---

         remapped_node_map = {}
         current_input_indices_kept_abs = self.pruning_map.get("global_in", torch.tensor([], dtype=torch.long)) # Indices from original input that are kept after the first slice

         # Simulate the pipeline stage by stage, defining what's KEPT after each slice point
         # based on what's REQUIRED by the *next* stage from the *original* map.

         # Stages are the layers whose OUTPUT we are slicing (except first layer)
         # original_indices_map (self.pruning_map) key maps:
         # 'global_in' -> required abs indices FROM global input for block0.conv1
         # 'temporal_blocks.0.conv1' -> required abs indices FROM block0.conv1 output for block0.conv2
         # 'temporal_blocks.0.conv2' -> required abs indices FROM block0.conv2 output for block1.conv1 (Block 0 out)
         # ... etc.

         # Initial state before any processing: all original input indices
         # Let's assume original_input_length is stored/available, perhaps calculate here
         # using the block configs? Calculate based on output len and reverse flow?
         # OR just load it from calculated_original_lengths.json
         try:
             with open("calculated_original_lengths.json", "r") as f:
                  orig_lens_data = json.load(f)
             original_input_length = orig_lens_data.get('original_input_length')
             if original_input_length is None:
                 raise ValueError("original_input_length not found in calculated_original_lengths.json")
         except Exception as e:
              print(f"‼️ ERROR loading original_input_length: {e}"); return # Cannot proceed

         initial_all_abs_input_indices = torch.arange(original_input_length, dtype=torch.long)

         # First slice point: "global_in" (input tensor)
         # Original indices needed FROM global_in for the next layer (block0.conv1)
         original_indices_needed_from_global_in = self.pruning_map.get("global_in", torch.tensor([], dtype=torch.long))
         # The indices to select AT THIS stage are these needed indices, relative to the current tensor (the full input).
         # Re mapping needed: Remap original_indices_needed_from_global_in relative to initial_all_abs_input_indices
         # This is trivial: remap([168..832], [0..999]) = [168..832]. The indices for DirectSlice are the absolute ones themselves.
         remapped_indices_for_global_in = self.remap_indices(original_indices_needed_from_global_in, initial_all_abs_input_indices)

         remapped_node_map["global_in"] = remapped_indices_for_global_in.tolist()
         # Update kept_indices_absolute based on this first slice
         current_absolute_indices_kept = initial_all_abs_input_indices[remapped_indices_for_global_in] # These are the original input indices that made it through the first slice.

         # Simulate Block 0, Conv 1 Forward
         # Need to know which original *outputs* Layer X produces when run on inputs defined by current_absolute_indices_kept
         # Function: get_output_indices_from_input_indices(input_kept_abs, layer_config, original_output_length)

         def get_output_indices_from_input_indices(input_kept_abs, layer_name, original_output_length, layers_config_map):
             if input_kept_abs is None or input_kept_abs.numel() == 0:
                  return torch.tensor([], dtype=torch.long)

             config = layers_config_map.get(layer_name)
             if not config:
                  print(f"Warning: Config not found for layer {layer_name} in get_output_indices_from_input_indices.")
                  return torch.tensor([], dtype=torch.long)

             kernel_size = config['kernel_size']
             dilation = config['dilation']
             stride = config['stride']

             # Map absolute input indices to their relative positions IN THE CURRENT TENSOR (0, 1, 2...)
             # Assume input_kept_abs is already sorted
             relative_input_positions = torch.arange(input_kept_abs.numel(), dtype=torch.long)

             # Simulate applying the kernel (D=1 here, assuming remapped D is 1 for computation)
             # If kernel starts at relative position `rel_start`, it covers `rel_start` to `rel_start + K - 1`.
             # What range of *original* input indices does this relative window correspond to?
             # `input_kept_abs[rel_start + k * 1]` corresponds to original input.
             # We need to find which original *output* indices are produced by these original *input* indices.

             # Let's invert: which output indices COULD this layer produce if its *full original input* was used? Original output indices 0 to OriginalOutputLen - 1.
             # Which of these original output indices map back to the input indices we kept (`input_kept_abs`)?

             # This path feels complex again. Let's simplify the definition of the iterative method.

             # The core idea: The *values* in the tensor *at index i* after slicing should correspond to
             # the *values* from the *original full tensor* at a specific index calculated by traversing
             # the cumulative slicing decisions.

             # Okay, back to the remapping simulation as initially intended.
             # The error might have been in how `kept_indices_absolute` was used or updated.

             # Let's rethink the state: `current_absolute_indices_represented`: This tensor
             # represents the *absolute original indices* that are currently *at relative positions 0, 1, 2...*
             # in the tensor `x`.

             return self._generate_remapped_map(self.pruning_map, self.original_layer_configs, original_input_length) # Call the generation logic


    # --- Generate Remapped Map Function (Based on earlier simulation logic) ---
    # This needs to be a method of the class to access pruning_map etc.
    # Let's refactor the standalone script logic into this method.
    def _generate_remapped_map(self, original_abs_map, layers_config_map, original_input_length):
        """
        Generates the remapped indices map by simulating forward slicing.
        Original logic from the separate remapping script.
        """
        # Convert map values to tensors just before starting, if they aren't already
        current_original_abs_map_tensors = {}
        for key, value in original_abs_map.items():
             if isinstance(value, list):
                  current_original_abs_map_tensors[key] = torch.tensor(sorted(list(set(value))), dtype=torch.long)
             elif isinstance(value, torch.Tensor): # Assume already converted
                 current_original_abs_map_tensors[key] = value
             else:
                  current_original_abs_map_tensors[key] = torch.tensor([], dtype=torch.long) # Treat missing/invalid as empty


        remapped_node_map = {}
        device = torch.device('cpu') # Perform remapping calculation on CPU

        # --- Stage 0: Input ---
        # The absolute indices kept are those specified by 'global_in' from original_abs_map
        original_indices_needed_from_global_in = current_original_abs_map_tensors.get("global_in", torch.tensor([], dtype=torch.long))
        current_absolute_indices_represented = original_indices_needed_from_global_in.to(device) # These are the first indices kept.

        # The remapped indices for "global_in" slice are just their ranks if 0-based from start
        # Assume original input indices are 0..original_input_length-1
        initial_all_abs_input_indices = torch.arange(original_input_length, dtype=torch.long, device=device)

        # Remap required indices relative to the full input tensor (indices 0 .. original_input_length-1)
        remapped_indices_for_global_in = self.remap_indices(original_indices_needed_from_global_in.to(device), initial_all_abs_input_indices)
        remapped_node_map["global_in"] = remapped_indices_for_global_in.tolist()
        # After the global_in slice, the absolute original indices that are now present are precisely `original_indices_needed_from_global_in`
        current_absolute_indices_represented = original_indices_needed_from_global_in.to(device)


        # Define execution flow order based on the mapping keys
        # These keys map to layers whose OUTPUT we are slicing
        slice_keys_in_execution_order = [
             'temporal_blocks.0.conv1', # Output of this layer is sliced
             'temporal_blocks.0.conv2', # Output of this layer is sliced
             'temporal_blocks.1.conv1', # Output of this layer is sliced
             'temporal_blocks.1.conv2', # Output of this layer is sliced
             'temporal_blocks.2.conv1', # Output of this layer is sliced
             'temporal_blocks.2.conv2', # Output of this layer is sliced (final selection)
        ]

        # Need to map a layer name (output) to the layer name that takes it as INPUT for calculating influence.
        # Example: 'temporal_blocks.0.conv1' output goes to 'temporal_blocks.0.conv2'.
        # We need 'block configs' to map this.

        layer_output_to_input_layer = { # Maps the key for slicing point -> config key for the LAYER *using* that sliced output
             'temporal_blocks.0.conv1': 'temporal_blocks.0.conv2', # Output of conv1 in block0 is input to conv2 in block0
             'temporal_blocks.0.conv2': 'temporal_blocks.1.conv1', # Output of conv2 in block0 is input to conv1 in block1
             'temporal_blocks.1.conv1': 'temporal_blocks.1.conv2', # Output of conv1 in block1 is input to conv2 in block1
             'temporal_blocks.1.conv2': 'temporal_blocks.2.conv1', # Output of conv2 in block1 is input to conv1 in block2
             'temporal_blocks.2.conv1': 'temporal_blocks.2.conv2', # Output of conv1 in block2 is input to conv2 in block2
             # The output of temporal_blocks.2.conv2 is the final network output before FC/reshape.
             # Its required indices are 'block_output' in the original_abs_map.
             'temporal_blocks.2.conv2': 'FINAL_OUTPUT' # Special key indicating no more Conv layers
        }


        for slice_point_key in slice_keys_in_execution_order:
            # This key is like 'temporal_blocks.0.conv1', it means slice *after* this layer's output
            # Original absolute indices needed from the *original* output of this layer
            original_abs_indices_needed_from_this_output = current_original_abs_map_tensors.get(slice_point_key, torch.tensor([], dtype=torch.long))

            # Simulate the layer(s) *before* this slice point to figure out what absolute original indices
            # *are actually produced* in the current tensor 'x' before this slice point.
            # This simulation logic is complex and needs the full pipeline path...

            # Alternative simpler simulation logic:
            # `current_absolute_indices_represented` tracks which original *INPUT* indices survived up to this point.
            # The output of a layer (Conv, BN, etc.) on an input derived from these `current_absolute_indices_represented`
            # will produce outputs that correspond to a *different set* of absolute original indices from its *output space*.
            # We need a function to map: `abs_input_indices_kept -> set of abs output indices produced`.
            # `get_abs_output_indices_produced(abs_input_indices_kept, layer_name, original_output_length)`.

            # This get_abs_output_indices_produced function seems tricky to implement accurately
            # for chains of Conv/BN on sparse inputs.

            # Let's step back. The premise was:
            # Slice A (input) -> Conv1 (orig D) -> Slice B -> Conv2 (D=1) -> Slice C -> Conv3 (D=1) -> ...
            # The indices for Slice B are remapped based on original indices needed from Conv1's original output.
            # For this to work, Conv1 operating on Slice A *must produce* a tensor whose *values* at indices X, Y, Z...
            # correspond to the original values at needed indices P, Q, R... from original Conv1 output.

            # What if the "remap_indices" logic is applied not to original->sliced indices, but to original_output->currently_available_output_indices?

            # Let's simplify again: Trust that the *intended* input to the slicer at `slice_point_key` *should* conceptually be
            # a tensor containing the absolute original values corresponding to the original output space of that layer,
            # but filtered by *previous* slices. How many such original output values are still conceptually 'active'?

            # The set of absolute original indices that are represented by the current tensor is the cumulative intersection.
            # After Slice @ S_i, `abs_indices_kept_i = abs_indices_kept_{i-1} intersect abs_indices_required_by_Si_orig_map`

            # Let's recalculate `current_absolute_indices_represented` stage by stage based on INTERSECTION
            # Initial: `current_absolute_indices_represented` = `global_in` required abs indices.
            # Before L0.conv1 slice: `current_absolute_indices_represented` is `global_in` indices.
            # Layer L0.conv1 operates on these inputs -> produces outputs...
            # This output needs to be sliced using indices required from its original output ('temporal_blocks.0.conv1').
            # Let `needed_outputs_L1_abs = self.pruning_map['temporal_blocks.0.conv1']`
            # `current_absolute_indices_represented` becomes `intersect(original_input_indices_needed_for_L1_outputs_in_needed_outputs_L1_abs)`

            # This looks more like a forward-propagating set intersection (pruning) logic.

            # Let's trace which *original* output indices of layer L survive based on needed_outputs_L+1_abs
            # original_outputs_kept_L6_abs = pruning_map['temporal_blocks.2.conv2'] # [84]
            # original_outputs_kept_L5_abs = intersect(pruning_map['temporal_blocks.2.conv1'], outputs_of_L5_needed_for_L6_outputs_kept)
            # ... this is complex intersection tracing forward.

            # Final attempt at remapped map generation logic within __init__
            # This re-implements the `create_remapped_node_index_map` simulation

            print("Generating remapped index map internally...")

            # --- Need the Absolute Corrected Map (self.pruning_map) ---
            # Already loaded and in tensor format

            remapped_map = {}
            device_calc = torch.device('cpu') # Use CPU for map generation calculations

            # Map layer key -> indices needed from *original* output space of this layer
            abs_required_outputs_map = self.pruning_map # Using self.pruning_map for original abs required indices


            # Step 1: Global Input Slice
            # Required from original input (key 'global_in') -> indices to select AT the input stage
            # These ARE the first absolute indices kept.
            required_from_global_in_abs = abs_required_outputs_map.get('global_in', torch.tensor([], dtype=torch.long)).to(device_calc)
            remapped_map['global_in'] = self.remap_indices(required_from_global_in_abs, torch.arange(original_input_length, dtype=torch.long, device=device_calc)).tolist()
            current_abs_indices_in_current_tensor = required_from_global_in_abs # Absolute indices from original input currently *in* the tensor after input slice

            # --- Simulate Subsequent Stages ---
            # Need to iterate through layers in execution order, mapping outputs back to inputs
            # based on original configs, then taking intersection with kept indices.

            # Layer keys in execution order: temporal_blocks.i.convj
            block_execution_order = [
                 ('temporal_blocks.0', block.conv1, block.conv2, self.original_layer_configs.get('temporal_blocks.0.conv1'), self.original_layer_configs.get('temporal_blocks.0.conv2')) for block in self.temporal_blocks[:1] # Block 0
            ] + [
                 (f'temporal_blocks.{i}', self.temporal_blocks[i].conv1, self.temporal_blocks[i].conv2, self.original_layer_configs.get(f'temporal_blocks.{i}.conv1'), self.original_layer_configs.get(f'temporal_blocks.{i}.conv2')) for i in range(1, len(self.temporal_blocks)) # Blocks 1 and 2
            ]

            original_layer_key_order = [ # The keys used in original_abs_map
                 'global_in',
                 'temporal_blocks.0.conv1',
                 'temporal_blocks.0.conv2',
                 'temporal_blocks.1.conv1',
                 'temporal_blocks.1.conv2',
                 'temporal_blocks.2.conv1',
                 'temporal_blocks.2.conv2', # This is the final slice point
             ]

            # We need a way to get original *input* indices from original *output* indices for a given layer config
            def get_abs_input_from_abs_output(output_indices_abs, layer_config):
                if output_indices_abs is None or output_indices_abs.numel() == 0:
                     return torch.tensor([], dtype=torch.long, device=output_indices_abs.device if output_indices_abs is not None else device_calc)

                kernel_size = layer_config['kernel_size']
                dilation = layer_config['dilation']
                stride = layer_config['stride']

                input_indices = []
                for out_idx in output_indices_abs.tolist():
                     # output_idx = (input_idx - dilation*(kernel-1)) / stride --> input_idx = output_idx * stride + dilation*(kernel-1) ? No, simpler.
                     # output_idx = (input_idx_start_of_window - dilation*(kernel-1)) / stride + 1 - 1 -- simplified output index formula based on input index
                     # Let `in_pos` be an index in the input window `[in_pos_start .. in_pos_end]`. Output `out_idx` is related to `in_pos_start` by `out_idx * stride = in_pos_start`.
                     # The input window that produces output `out_idx` is `[out_idx * stride + k * dilation for k in 0..kernel_size-1]`

                     input_start_abs = out_idx * stride
                     influenced_inputs = [input_start_abs + k * dilation for k in range(kernel_size)]
                     input_indices.extend(influenced_inputs)

                # These are absolute indices from the layer's *direct* input.
                return torch.tensor(sorted(list(set(input_indices))), dtype=torch.long, device=output_indices_abs.device if output_indices_abs is not None else device_calc)


            # Iterate through the layers where slicing happens *after* their output
            layers_with_output_slices_keys = list(abs_required_outputs_map.keys()) # This includes 'global_in' and all conv outputs
            layers_with_output_slices_keys.remove('global_in') # Global input slice handled


            # To simulate correctly, we need to know which absolute *original* input indices correspond
            # to the *actual tensor* being produced at each stage.
            # `current_abs_indices_in_current_tensor` keeps track of this.

            # Let's try propagating the *set of kept original input indices* forward.
            current_abs_input_indices_kept_overall = current_abs_indices_in_current_tensor.clone() # Start with global_in slice results


            # Need original layer sequence + configs
            full_layer_sequence_configs = [
                ('temporal_blocks.0.conv1', self.original_layer_configs.get('temporal_blocks.0.conv1')),
                ('temporal_blocks.0.conv2', self.original_layer_configs.get('temporal_blocks.0.conv2')),
                ('temporal_blocks.1.conv1', self.original_layer_configs.get('temporal_blocks.1.conv1')),
                ('temporal_blocks.1.conv2', self.original_layer_configs.get('temporal_blocks.1.conv2')),
                ('temporal_blocks.2.conv1', self.original_layer_configs.get('temporal_blocks.2.conv1')),
                ('temporal_blocks.2.conv2', self.original_layer_configs.get('temporal_blocks.2.conv2')),
            ]

            for layer_key, layer_config in full_layer_sequence_configs:
                # layer_key is like 'temporal_blocks.0.conv1'
                # This layer operates on input derived from current_abs_input_indices_kept_overall

                # Which *absolute original output indices* are produced by THIS layer operating
                # on input corresponding to `current_abs_input_indices_kept_overall`?
                # Function: get_output_indices_from_input_indices(input_kept_abs, layer_config)

                def get_output_indices_produced_by_kept_input(input_kept_abs, layer_config, original_input_length_to_this_layer, original_output_length_of_this_layer):
                     if input_kept_abs is None or input_kept_abs.numel() == 0:
                         return torch.tensor([], dtype=torch.long)

                     kernel_size = layer_config['kernel_size']
                     dilation = layer_config['dilation']
                     stride = layer_config['stride']

                     produced_output_indices_abs = set()
                     # Iterate through all possible starting positions of the kernel on the original input space
                     # that overlap with input_kept_abs.
                     # Find min/max indices in input_kept_abs to limit search space
                     if input_kept_abs.numel() == 0: return torch.tensor([], dtype=torch.long)

                     min_abs_input = torch.min(input_kept_abs)
                     max_abs_input = torch.max(input_kept_abs)

                     # Possible original output indices `j`
                     min_possible_output_j = 0
                     max_possible_output_j = original_output_length_of_this_layer - 1

                     for out_j in range(min_possible_output_j, max_possible_output_j + 1):
                         # Calculate the original input window indices for this original output j
                         original_input_window_indices = [out_j * stride + k * dilation for k in range(kernel_size)]

                         # Check if this input window has any overlap with the kept input indices
                         # If ANY index in the window is in input_kept_abs, then this output IS potentially computed
                         window_overlaps = False
                         for win_idx in original_input_window_indices:
                              # Fast check for presence in sorted input_kept_abs
                              # Check if win_idx is within range AND is actually in the kept list
                              if win_idx >= min_abs_input and win_idx <= max_abs_input:
                                   pos = torch.searchsorted(input_kept_abs, win_idx)
                                   if pos < input_kept_abs.numel() and input_kept_abs[pos] == win_idx:
                                        # Found at least one needed index in the kept inputs
                                        # Is it enough? Valid padding means *all* indices needed must be present for that specific kernel position.
                                        # If ANY index is missing from input_kept_abs within this window, this output is NOT produced correctly.
                                        window_fully_kept = True
                                        for inner_win_idx in original_input_window_indices:
                                              pos_inner = torch.searchsorted(input_kept_abs, inner_win_idx)
                                              if pos_inner >= input_kept_abs.numel() or input_kept_abs[pos_inner] != inner_win_idx:
                                                   window_fully_kept = False
                                                   break # This window cannot be computed correctly
                                        if window_fully_kept:
                                             produced_output_indices_abs.add(out_j)
                                             break # Move to next output j

                     return torch.tensor(sorted(list(produced_output_indices_abs)), dtype=torch.long)

            # Need original lengths map here as well
            try:
                 with open("calculated_original_lengths.json", "r") as f:
                      orig_lens_data = json.load(f)
            except Exception as e:
                  print(f"‼️ ERROR loading original lengths for remapped map gen: {e}"); return {}


            # Get input and output original lengths for this layer
            input_len_key = 'global_input' if layer_key == 'temporal_blocks.0.conv1' else layer_output_to_input_layer.get(layer_key) # Need mapping from current layer to the key of its input layer
            if input_len_key is None:
                 print(f"‼️ Warning: Cannot find input length key for layer {layer_key}.")
                 input_original_length_this_layer = None
            else:
                 input_original_length_this_layer = orig_lens_data.get(input_len_key)
                 if input_original_length_this_layer is None:
                      print(f"‼️ Warning: Original input length {input_len_key} not found in lengths file for layer {layer_key}.")

             # Original output length of this layer
            output_original_length_this_layer = orig_lens_data.get(layer_key.replace('.', '_') + '_output', None) # Example: temporal_blocks_0_conv1_output --> original_lengths_map['output_L1']


            produced_outputs_abs = get_output_indices_produced_by_kept_input(
                 current_abs_input_indices_kept_overall.to(device_calc), # Use the indices kept AFTER the previous slice
                 layer_config,
                 input_original_length_this_layer, # Pass original input length to this layer
                 output_original_length_this_layer # Pass original output length of this layer
            ).to(device_calc)


            # --- Intersect produced outputs with REQUIRED outputs from original_abs_map ---
            # Needed original outputs from THIS layer's output for the next step (key is layer_key itself)
            required_from_this_output_abs = abs_required_outputs_map.get(layer_key, torch.tensor([], dtype=torch.long)).to(device_calc)
            # The absolute original output indices that survive THIS stage's PRUNING are the intersection
            surviving_outputs_abs = torch.tensor(sorted(list(set(produced_outputs_abs.tolist()) & set(required_from_this_output_abs.tolist()))), dtype=torch.long, device=device_calc)
            # These `surviving_outputs_abs` are the absolute indices that will be *represented* by the tensor
            # *after* the slice corresponding to `layer_key`.
            # The REMAPPED indices to apply the slice: `surviving_outputs_abs` remapped relative to `produced_outputs_abs`.
            remapped_indices_for_this_slice = self.remap_indices(surviving_outputs_abs, produced_outputs_abs).to(device_calc)
            remapped_map[layer_key] = remapped_indices_for_this_slice.tolist()
            # --- Update the set of ABSOLUTE original *input* indices that are still kept ---
            # We need the absolute original *input* indices required to produce `surviving_outputs_abs`.
            # Use the reverse mapping again.
            input_indices_to_keep_for_next_stage_abs = get_abs_input_from_abs_output(surviving_outputs_abs, layer_config)
            # This set of input indices is intersected with the indices kept *before* this layer to update.
            # However, `get_abs_input_from_abs_output` already gets input from THIS layer.
            # Need to trace *this layer's* input back further... this chaining is where the manual sim failed.
            # Let's redefine current_absolute_indices_represented: it *is* the set of absolute input indices that survived.
            # And get_abs_input_from_abs_output gives us required input indices FOR A SINGLE LAYER.
            # We need to propagate this backward correctly.
            # Simplified tracking:
            # current_abs_input_indices_kept_overall starts with 'global_in' slice results.
            # AFTER layer N + slice: `current_abs_input_indices_kept_overall` = indices required for SURVIVING outputs of Layer N, traced back to ORIGINAL INPUT.
            # This looks like recalculating receptive fields dynamically.
            # Okay, maybe simpler: current_abs_indices_in_current_tensor is correct as ABSOLUTE original indices that survive to the current tensor.
            # After Layer L: The current tensor represents original Layer L output at `current_abs_indices_in_current_tensor`.
            # Required outputs from Layer L original map: `required_outputs_L_abs`.
            # Survive Layer L slice: `intersection(current_abs_indices_in_current_tensor, required_outputs_L_abs)`. This is the set of abs *L outputs* kept.
            # Update state: `current_abs_indices_in_current_tensor` = trace_back(surviving_outputs_L_abs, layer_config)
            # This feels like what I tried before and got lost.
            # --- FINAL ATTEMPT at simple remapped generation simulation ---
            # Trusting the previous standalone remapping script logic:
            # `current_absolute_indices_represented` = set of abs ORIGINAL INPUT indices currently represented
            # Each step: Apply a 'virtual layer + virtual slice'
            # A virtual layer maps indices `I -> O` based on its config
            # A virtual slice keeps `needed_O` indices, updates `current_absolute_indices_represented` based on *which I* map to `needed_O` within the currently represented I set.
            # Okay, the standalone `create_remapped_node_index_map` script is the most concrete attempt at simulating this chain of remapping.
            # Let's make sure *that* script is producing valid JSON, and that its remapping logic (using `remap_indices`) is correctly updating the `kept_indices_absolute` array representing original input indices.
            # If *that* script's final indices are empty, it indicates the strategy is pruning too aggressively.
            # If that script produces valid looking remapped indices, *then* THIS forward pass implementation (using those pre-calculated remapped indices) should work numerically.
            # Reverting _generate_remapped_map logic. TCN2d_inf_remap should load the PRE-CALCULATED remapped map.


        # This _forward_with_iterative_pruning logic relies on `self.remapped_pruning_map` which is loaded in __init__.
        return self._forward_with_iterative_pruning(x)

    # Ensure remap_indices is available within the class scope
    # It can be a static method or stand-alone
    @staticmethod
    def remap_indices(original_relevant_indices_tensor, sorted_kept_indices_tensor):
        """
        Remaps original indices to their new positions within a sliced tensor.
        (Same logic as previously refined remap_indices)
        """
        if original_relevant_indices_tensor is None or sorted_kept_indices_tensor is None or sorted_kept_indices_tensor.numel() == 0:
            return torch.tensor([], dtype=torch.long, device=original_relevant_indices_tensor.device if original_relevant_indices_tensor is not None else sorted_kept_indices_tensor.device if sorted_kept_indices_tensor is not None else torch.device('cpu'))


        # Ensure tensors are on the same device and LongTensor
        device = sorted_kept_indices_tensor.device
        original_relevant_indices_tensor = original_relevant_indices_tensor.to(device).long()
        sorted_kept_indices_tensor = sorted_kept_indices_tensor.to(device).long()


        # Find which original relevant indices are present in sorted_kept_indices
        # Use broadcasting comparison - fast intersection equivalent
        # Need to handle cases where one tensor is much larger than the other
        # Efficient intersection + getting indices: Use searchsorted

        # Find where each original_relevant_index would be inserted into sorted_kept_indices_tensor
        insertion_points = torch.searchsorted(sorted_kept_indices_tensor, original_relevant_indices_tensor)

        # Check if the element AT the insertion point actually matches the original_relevant index
        # Handle cases where insertion_points is out of bounds of sorted_kept_indices_tensor
        valid_insertion_mask = (insertion_points < sorted_kept_indices_tensor.numel()) & (insertion_points >= 0) # Ensure >= 0
        # Filter original_relevant_indices and their insertion points based on valid mask
        original_relevant_indices_valid = original_relevant_indices_tensor[valid_insertion_mask]
        insertion_points_valid = insertion_points[valid_insertion_mask]

        if original_relevant_indices_valid.numel() == 0:
             return torch.tensor([], dtype=torch.long, device=device)

        # Check if the value at the insertion point in kept matches the value we wanted
        # This identifies elements that were truly "kept" from the original_relevant list
        matches = (sorted_kept_indices_tensor[insertion_points_valid] == original_relevant_indices_valid)

        # The remapped indices are the insertion points corresponding to the matches
        remapped_indices = insertion_points_valid[matches]

        # Ensure uniqueness and sorting just in case (though should be from construction)
        # Unique op implies sorting
        # return torch.unique(remapped_indices).to(torch.long) # Unique implies sorting, but might break if input wasn't unique

        # The list of remapped indices (positions within the sorted_kept_indices_tensor)
        # If original_relevant_indices_tensor was unique and sorted, and sorted_kept_indices_tensor is unique and sorted,
        # then remapped_indices should be unique and sorted if matches.
        return remapped_indices.to(torch.long)



# --- Add the __call__ method to TCN2d_inf_iterative_pruning ---
    def __call__(self, x):
        return self.forward(x)

# --- Main Comparison Script Logic (Same as before) ---

# Import or define TCN2d (your original) and TemporalBlock2d if needed by TCN2d

# --- Configuration ---
CHECKPOINT_PATH = "/home/eveneiha/finn/workspace/ml/model/tcn_model_v41.pth" # !!! Update this path
NUM_OUTPUTS = 5 # Your model's output size

# Define block parameters matching the original model *exactly* (used for training)
block_configs_orig = [
    {'n_inputs': 1, 'n_outputs': 4, 'kernel_size': 9, 'dilation': 1, 'stride': 2, 'dropout': 0.0}, # Dropout 0 for eval
    {'n_inputs': 4, 'n_outputs': 8, 'kernel_size': 9, 'dilation': 4, 'stride': 1, 'dropout': 0.0}, # Dropout 0 for eval
    {'n_inputs': 8, 'n_outputs': 16, 'kernel_size': 9, 'dilation': 16, 'stride': 1, 'dropout': 0.0},# Dropout 0 for eval
]

# Define a representative TEST_INPUT_LEN
# This must match the length used when running indices_corrected.py
# and the script that generates remapped_relevant_indices_per_model_layer.json
TEST_INPUT_LEN = 1000 # Or the actual length used
TEST_BATCH_SIZE = 1
TEST_INPUT_CHANNELS = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- --- IMPORTANT PREPARATION --- ---
# 1. Run your indices.py (corrected version using UNION) to generate:
#    "corrected_relevant_indices_per_layer.json"
#    "calculated_original_lengths.json" (Make sure indices.py saves this)
# 2. Run a SEPARATE script (based on create_remapped_node_index_map logic)
#    to load "corrected_relevant_indices_per_layer.json" and
#    generate "remapped_relevant_indices_per_model_layer.json"

# >>> Make SURE "remapped_relevant_indices_per_model_layer.json" file exists <<<
# and "corrected_relevant_indices_per_model_layer.json" exists (for TCN2d_inf_iterative_pruning init)
# and "calculated_original_lengths.json" exists (if needed by generation logic inside TCN class)


# --- Instantiate Models ---
print("Instantiating models...")
# Original model
# Assuming TCN2d class is defined and matches original model used for training/map generation
# Need to make sure it uses TemporalBlock2d matching your training (with original dropout value)
# And that for EVALUATION, dropout is set to 0, usually by model_original.eval().
block1 = TemporalBlock2d(1, 4, kernel_size=9, dilation=1,  stride=2,  dropout=0.05)
block2 = TemporalBlock2d(4, 8, kernel_size=9, dilation=4,  stride=1,  dropout=0.05)
block3 = TemporalBlock2d(8, 16, kernel_size=9, dilation=16,  stride=1,  dropout=0.05)

custom_blocks = [block1, block2, block3]

model_original = TCN2d(custom_blocks, NUM_OUTPUTS).to(device)


# Pruned/Sliced model using iterative pruning logic
# Pass the path to the CORRECTED absolute map file to __init__
REMAPPPED_MAP_FILE = "remapped_relevant_indices_per_model_layer.json"
CORRECTED_ABS_MAP_FILE = "corrected_relevant_indices_per_model_layer.json"

model_pruned_sliced = TCN2d_inf_iterative_pruning(
    block_configs_orig, NUM_OUTPUTS, CORRECTED_ABS_MAP_FILE
).to(device)

# Explicitly load the REMAAPED map in the instantiated model.
# Or adapt __init__ to load REMAPPED_MAP_FILE
if model_pruned_sliced.pruning_map: # If initial load succeeded
     try:
          with open(REMAPPPED_MAP_FILE, "r") as f:
             remapped_pruning_data = json.load(f)
          model_pruned_sliced.remapped_pruning_map = {} # Overwrite the abs map placeholder
          for key, value in remapped_pruning_data.items():
               if isinstance(value, list):
                    # Convert remapped indices to LongTensor
                    model_pruned_sliced.remapped_pruning_map[key] = torch.tensor(value, dtype=torch.long).to(device)
               else:
                   model_pruned_sliced.remapped_pruning_map[key] = None
          print(f"Loaded REMAAPED pruning map from {REMAPPPED_MAP_FILE}")
          # print(f"Remapped map keys: {model_pruned_sliced.remapped_pruning_map.keys()}")

     except FileNotFoundError:
          print(f"‼️ ERROR: REMAAPED map file not found at {REMAPPPED_MAP_FILE}.")
          print("Pruned/Sliced model will likely fail or use empty slices.")
          model_pruned_sliced.remapped_pruning_map = {} # Clear to indicate problem
     except json.JSONDecodeError:
          print(f"‼️ ERROR: REMAAPED map file at {REMAPPPED_MAP_FILE} is invalid JSON.")
          model_pruned_sliced.remapped_pruning_map = {} # Clear to indicate problem


# --- Load Trained Weights ---
print(f"Loading state dict from: {CHECKPOINT_PATH}")
try:
    # Load the ENTIRE checkpoint file
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    # Assuming the model state dict is stored under the key 'model_state_dict'
    # Access the dictionary containing only the model's weights
    model_state_dict = checkpoint['model_state_dict']

    # --- Load into the original model ---
    # Use the model_state_dict, which now contains the correct keys like temporal_blocks...
    model_original.load_state_dict(model_state_dict) # Use strict=True implicitly, will error if mismatch still exists
    print("Original model state dict loaded successfully.")

    # --- Load into the pruned/sliced model ---
    # Load the same model_state_dict into the pruned/sliced model
    # Use strict=False here as before to ignore slicer params etc.
    model_pruned_sliced.load_state_dict(model_state_dict, strict=False)
    print("Pruned/Sliced model state dict loaded successfully (using strict=False).")


except FileNotFoundError:
    print(f"ERROR: Checkpoint file not found at {CHECKPOINT_PATH}")
    print("Skipping accuracy comparison.")
    exit()
except KeyError as e:
     print(f"ERROR: Could not find the model state dictionary key in checkpoint file.")
     print(f"Expected key 'model_state_dict' not found. Check your checkpoint file structure.")
     print(f"Available keys in checkpoint: {checkpoint.keys() if 'checkpoint' in locals() else 'N/A'}")
     print(f"Skipping accuracy comparison.")
     exit()

except Exception as e:
    print(f"ERROR: Failed during state dict loading after finding model_state_dict: {e}")
    print("Skipping accuracy comparison.")
    exit()


# --- Set to Evaluation Mode ---
# Dropout layers and BatchNorm behavior change in eval mode
model_original.eval()
model_pruned_sliced.eval()
print("Models set to evaluation mode.")

# --- Create Dummy Input ---
dummy_input = torch.randn(TEST_BATCH_SIZE, TEST_INPUT_CHANNELS, TEST_INPUT_LEN, 1).to(device)
print(f"Created dummy input with shape: {dummy_input.shape}")

# --- Run Inference ---
print("Running inference...")
with torch.no_grad():
    output_original = model_original(dummy_input)

    # Run the iteratively pruned/sliced model
    output_pruned_sliced = model_pruned_sliced(dummy_input)


# --- Compare Outputs ---
print("\n--- Comparison Results ---")
print("Original Model Output Shape:", output_original.shape)
print("Pruned/Sliced Model Output Shape:", output_pruned_sliced.shape)


# Check for empty outputs first (if a slice resulted in an empty tensor)
if output_pruned_sliced.numel() == 0 and output_original.numel() > 0:
     print("\n‼️ WARNING: Pruned/Sliced model produced an empty output!")
elif output_original.shape != output_pruned_sliced.shape:
    print("\n‼️ WARNING: Output shapes differ!")
    # You might need debugging print statements in TCN2d_inf_iterative_pruning forward to see why final shape is wrong
else:
    # Calculate difference only if shapes match and are non-empty
    abs_diff = torch.abs(output_original - output_pruned_sliced)
    sum_abs_diff = abs_diff.sum().item()
    max_abs_diff = abs_diff.max().item()

    # Use torch.allclose for numerical comparison
    # Adjust atol (absolute tolerance) and rtol (relative tolerance) as needed
    # Iterative slicing/dilation change might introduce more deviation than simple end-slicing.
    tolerance_atol = 1e-5 # Absolute tolerance - potentially increase slightly
    tolerance_rtol = 1e-4 # Relative tolerance - potentially increase slightly
    outputs_are_close = torch.allclose(output_original, output_pruned_sliced, atol=tolerance_atol, rtol=tolerance_rtol)

    print(f"\nSum of Absolute Differences: {sum_abs_diff:.8f}")
    print(f"Maximum Absolute Difference: {max_abs_diff:.8f}")
    print(f"Outputs are close (atol={tolerance_atol}, rtol={tolerance_rtol}): {outputs_are_close}")

    if outputs_are_close:
        print("\n✅ Accuracy Preserved (Numerically Close).")
        print("Proceeding with FINN export and potential custom transformations for this architecture.")
        # --- Export model_pruned_sliced to ONNX here if successful ---
        # from qonnx.core.modelwrapper import ModelWrapper
        # from qonnx.util.cleanup import cleanup
        # dummy_input_for_export = torch.randn(1, TEST_INPUT_CHANNELS, TEST_INPUT_LEN, 1).to(device)
        # export_onnx_path = './onnx/tcn_iterative_pruned.onnx'
        # torch.onnx.export(model_pruned_sliced, dummy_input_for_export, export_onnx_path, verbose=False, opset_version=11) # Opset might need adjusting
        # cleanup(export_onnx_path, out_file=export_onnx_path)
        # print(f"Model exported to {export_onnx_path}")

    else:
        print("\n❌ Accuracy NOT Preserved. Outputs differ significantly.")
        print("This iterative slicing/dilation change strategy did NOT preserve accuracy numerically in PyTorch.")
        print("Debugging needed: check index calculations, remapping logic, dilation changes, quantizers, original layer order.")
        print("It's possible this strategy is fundamentally incompatible despite the mathematical premise for certain parameters/inputs.")