# import torch
# import torch.nn as nn
# import torch.nn.utils.parametrize as parametrize
# from brevitas import nn as qnn
# from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat

# class TemporalBlock2d(nn.Module):
#     # ... (TemporalBlock2d class remains the same as before) ...
#     def __init__(self, n_inputs, n_outputs, kernel_size, dilation, stride, dropout=0.05):
#         super(TemporalBlock2d, self).__init__()
#         self.kernel_size = kernel_size
#         self.dilation = dilation
#         self.stride = stride

#         self.conv1 = qnn.QuantConv2d(
#             n_inputs, n_outputs,
#             kernel_size=(kernel_size, 1),
#             stride=(stride, 1),
#             padding=(0, 0),  # no padding – only compute valid outputs
#             dilation=(dilation, 1),
#             weight_quant=Int8WeightPerTensorFloat,
#             input_quant=Int8ActPerTensorFloat,
#             weight_bit_width=8,
#             act_bit_width=8,
#             bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(n_outputs)
#         self.dropout1 = nn.Dropout(dropout)

#         self.conv2 = qnn.QuantConv2d(
#             n_outputs, n_outputs,
#             kernel_size=(kernel_size, 1),
#             stride=(1, 1),
#             padding=(0, 0),
#             dilation=(dilation, 1),
#             weight_quant=Int8WeightPerTensorFloat,
#             input_quant=Int8ActPerTensorFloat,
#             weight_bit_width=8,
#             act_bit_width=8,
#             bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(n_outputs)
#         self.dropout2 = nn.Dropout(dropout)

#         self.relu_after_conv = qnn.QuantReLU(return_quant_tensor=True)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.dropout1(x)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu_after_conv(x).value
#         x = self.dropout2(x)
#         return x


# block1 = TemporalBlock2d(1, 4, kernel_size=9, dilation=1,  stride=2,  dropout=0.05)
# block2 = TemporalBlock2d(4, 8, kernel_size=9, dilation=4,  stride=1,  dropout=0.05)
# block3 = TemporalBlock2d(8, 16, kernel_size=9, dilation=16,  stride=1,  dropout=0.05)

# input_tensor = torch.randn(1, 1, 1000, 1) # [N, C_in, L_in, 1] -  assuming temporal dimension is L

# output_block1 = block1(input_tensor)
# output_block2 = block2(output_block1)
# output_block3 = block3(output_block2)


# target_output_neuron_index = 84 # We are interested in output neuron number 84 in the last conv of block3

# # --- Analysis ---

# def get_input_sample_influence(block, output_index, layer_name, input_length):
#     """ ... (get_input_sample_influence function remains the same) ... """
#     if layer_name == 'conv1':
#         conv_layer = block.conv1
#         stride = block.stride
#         dilation = block.dilation
#         kernel_size = block.kernel_size
#     elif layer_name == 'conv2':
#         conv_layer = block.conv2
#         stride = 1 # Always 1 for conv2 in this setup
#         dilation = block.dilation # Dilation same for both conv layers in block
#         kernel_size = block.kernel_size
#     else:
#         raise ValueError("layer_name must be 'conv1' or 'conv2'")


#     influenced_input_samples = []
#     for kernel_index in range(kernel_size):
#         input_sample_index = output_index * stride + kernel_index * dilation
#         influenced_input_samples.append(input_sample_index)

#     # Remove indices that are out of bounds
#     valid_input_samples = [idx for idx in influenced_input_samples if 0 <= idx < input_length] # Check against input_length now
#     return sorted(list(set(valid_input_samples))) # Remove duplicates and sort


# # --- Traceback for block3's conv2 output 84 ---

# relevant_indices_per_layer = {} # Dictionary to store relevant indices for each layer
# non_relevant_indices_per_layer = {} # Dictionary to store non-relevant indices for each layer

# current_output_index = target_output_neuron_index
# relevant_input_indices = [current_output_index] # Start with the target output neuron

# print(f"Target output neuron in block3.conv2: {target_output_neuron_index}")

# relevant_indices_per_layer['block_output'] = [target_output_neuron_index] # Store relevant indices for block3 output
# non_relevant_indices_per_layer['block_output'] = [i for i in range(output_block3.shape[2]) if i != target_output_neuron_index] # Store non-relevant indices for block3 output

# #print(non_relevant_indices_per_layer['block_output'])

# # Block 3 - conv2
# input_length_block3_conv2 = output_block3.shape[2] + (block3.conv1.dilation[0] * (block3.conv1.kernel_size[0] - 1)) # Input length to block3.conv2 is output length of block3.conv1
# output_length_block3_conv2 = output_block3.shape[2]
# block3_conv2_influenced_indices = []
# for output_index in relevant_input_indices: # In first iteration, this is just [84]
#     indices = get_input_sample_influence(block3, output_index, 'conv2', input_length_block3_conv2) # Input length to conv2
#     block3_conv2_influenced_indices.extend(indices)
# block3_conv2_influenced_indices = sorted(list(set(block3_conv2_influenced_indices)))
# print(f"\nBlock3 - conv2: Output neuron {target_output_neuron_index} is influenced by input samples from block3 - conv1 at indices: {block3_conv2_influenced_indices}")
# print(f"Block3 - conv2: Total output samples: {output_length_block3_conv2}, Total influencing input samples: {len(block3_conv2_influenced_indices)}")
# relevant_indices_per_layer['block3_conv2'] = block3_conv2_influenced_indices # Store indices
# non_relevant_indices_per_layer['block3_conv2'] = [i for i in range(input_length_block3_conv2) if i not in block3_conv2_influenced_indices] # Store non-relevant indices
# relevant_input_indices = block3_conv2_influenced_indices # Update for next layer


# # Block 3 - conv1
# input_length_block3_conv1 = output_block2.shape[2] # Input length to block3.conv1 is output length of block1
# output_length_block3_conv1 = output_block3.shape[2] + (block3.conv1.dilation[0] * (block3.conv1.kernel_size[0] - 1))
# block3_conv1_influenced_indices = []
# for output_index in relevant_input_indices:
#     indices = get_input_sample_influence(block3, output_index, 'conv1', input_length_block3_conv1) # Input length to conv1
#     block3_conv1_influenced_indices.extend(indices)
# block3_conv1_influenced_indices = sorted(list(set(block3_conv1_influenced_indices)))
# print(f"Block3 - conv1: Influenced by input samples from block2 at indices: {block3_conv1_influenced_indices}")
# print(f"Block3 - conv1: Total output samples: {output_length_block3_conv1}, Total influencing input samples: {len(block3_conv1_influenced_indices)}")
# relevant_indices_per_layer['block3_conv1'] = block3_conv1_influenced_indices # Store indices
# non_relevant_indices_per_layer['block3_conv1'] = [i for i in range(input_length_block3_conv1) if i not in block3_conv1_influenced_indices] # Store non-relevant indices
# relevant_input_indices = block3_conv1_influenced_indices


# # Block 2 - conv2
# input_length_block2_conv2 = output_block2.shape[2] + (block2.conv1.dilation[0] * (block2.conv1.kernel_size[0] - 1)) # Input length to block2.conv2 is output length of block2.conv1
# output_length_block2_conv2 = output_block2.shape[2]
# block2_conv2_influenced_indices = []
# for output_index in relevant_input_indices:
#     indices = get_input_sample_influence(block2, output_index, 'conv2', input_length_block2_conv2) # Input length to conv2
#     block2_conv2_influenced_indices.extend(indices)
# block2_conv2_influenced_indices = sorted(list(set(block2_conv2_influenced_indices)))
# print(f"\nBlock2 - conv2: Influenced by input samples from block2 - conv1 at indices: {block2_conv2_influenced_indices}")
# print(f"Block2 - conv2: Total output samples: {output_length_block2_conv2}, Total influencing input samples: {len(block2_conv2_influenced_indices)}")
# relevant_indices_per_layer['block2_conv2'] = block2_conv2_influenced_indices # Store indices
# non_relevant_indices_per_layer['block2_conv2'] = [i for i in range(input_length_block2_conv2) if i not in block2_conv2_influenced_indices] # Store non-relevant indices
# relevant_input_indices = block2_conv2_influenced_indices

# # Block 2 - conv1
# input_length_block2_conv1 = output_block1.shape[2] # Input length to block2.conv1 is input tensor length
# output_length_block2_conv1 = output_block2.shape[2] + (block2.conv1.dilation[0] * (block2.conv1.kernel_size[0] - 1))
# block2_conv1_influenced_indices = []
# for output_index in relevant_input_indices:
#     indices = get_input_sample_influence(block2, output_index, 'conv1', input_length_block2_conv1) # Input length to conv1
#     block2_conv1_influenced_indices.extend(indices)
# block2_conv1_influenced_indices = sorted(list(set(block2_conv1_influenced_indices)))
# print(f"Block2 - conv1: Influenced by input samples from block1 at indices: {block2_conv1_influenced_indices}")
# print(f"Block2 - conv1: Total output samples: {output_length_block2_conv1}, Total influencing input samples: {len(block2_conv1_influenced_indices)}")
# relevant_indices_per_layer['block2_conv1'] = block2_conv1_influenced_indices # Store indices
# non_relevant_indices_per_layer['block2_conv1'] = [i for i in range(input_length_block2_conv1) if i not in block2_conv1_influenced_indices] # Store non-relevant indices
# relevant_input_indices = block2_conv1_influenced_indices


# # Block 1 - conv2
# input_length_block1_conv2 = output_block1.shape[2] + (block1.conv1.dilation[0] * (block1.conv1.kernel_size[0] - 1))# Input length to block1.conv2 is output length of block1.conv1
# output_length_block1_conv2 = output_block1.shape[2]  #Output length is same as input length for conv2 in block1
# block1_conv2_influenced_indices = []
# for output_index in relevant_input_indices:
#     indices = get_input_sample_influence(block1, output_index, 'conv2', input_length_block1_conv2) # Input length to conv2
#     block1_conv2_influenced_indices.extend(indices)
# block1_conv2_influenced_indices = sorted(list(set(block1_conv2_influenced_indices)))

# #block1_conv2_influenced_indices = block1_conv2_influenced_indices[4:-4]
# print(f"\nBlock1 - conv2: Influenced by input samples from block1 - conv1 at indices: {block1_conv2_influenced_indices}")
# print(f"Block1 - conv2: Total output samples: {output_length_block1_conv2}, Total influencing input samples: {len(block1_conv2_influenced_indices)}")
# relevant_indices_per_layer['block1_conv2'] = block1_conv2_influenced_indices # Store indices
# non_relevant_indices_per_layer['block1_conv2'] = [i for i in range(input_length_block1_conv2) if i not in block1_conv2_influenced_indices] # Store non-relevant indices
# relevant_input_indices = block1_conv2_influenced_indices


# # Block 1 - conv1 (Original Input)
# input_length_block1_conv1 = input_tensor.shape[2] # Input length to block1.conv1 is input tensor length
# output_length_block1_conv1 = output_block1.shape[2] + (block1.conv1.dilation[0] * (block1.conv1.kernel_size[0] - 1)) 
# block1_conv1_influenced_indices = []
# for output_index in relevant_input_indices:
#     indices = get_input_sample_influence(block1, output_index, 'conv1', input_length_block1_conv1) # Input length to conv1
#     block1_conv1_influenced_indices.extend(indices)
# block1_conv1_influenced_indices = sorted(list(set(block1_conv1_influenced_indices)))
# print(f"Block1 - conv1: Influenced by input samples from the original input tensor at indices: {block1_conv1_influenced_indices}")
# print(f"Block1 - conv1: Total output samples: {output_length_block1_conv1}, Total influencing input samples: {len(block1_conv1_influenced_indices)}")
# relevant_indices_per_layer['block1_conv1'] = block1_conv1_influenced_indices # Store indices
# non_relevant_indices_per_layer['block1_conv1'] = [i for i in range(input_length_block1_conv1) if i not in block1_conv1_influenced_indices] # Store non-relevant indices
# relevant_input_indices = block1_conv1_influenced_indices

# print("\n--- Relevant Indices Per Layer ---")
# for layer_name, indices in relevant_indices_per_layer.items():
#     print(f"{layer_name}: {indices}")
#     if layer_name == 'block1_conv2':
#         print(f"Total relevant indices for {layer_name}: {len(indices)}")
    
# print("\n--- Non-Relevant Indices Per Layer ---")
# for layer_name, indices in non_relevant_indices_per_layer.items():
#     print(f"{layer_name}: {indices}")

# import json

# with open("relevant_indices_per_layer.json", "w") as f:
#     json.dump(relevant_indices_per_layer, f)
    
# with open("non_relevant_indices_per_layer.json", "w") as f:
#     json.dump(non_relevant_indices_per_layer, f)
    
    

    
    
    
# --- START OF FILE indices.py (Corrected Version) ---

import torch
import torch.nn as nn
from brevitas import nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
import json
from functools import reduce
import math # Use math.floor

class TemporalBlock2d(nn.Module):
    # ... (TemporalBlock2d class definition remains the same) ...
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, stride, dropout=0.05):
        super(TemporalBlock2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride

        self.conv1 = qnn.QuantConv2d(
            n_inputs, n_outputs,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(0, 0),  # no padding – only compute valid outputs
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
            stride=(1, 1),
            padding=(0, 0),
            dilation=(dilation, 1),
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu_after_conv(x).value
        x = self.dropout2(x)
        return x

# --- Corrected Analysis ---

# Define blocks (needed for their parameters)
block1 = TemporalBlock2d(1, 4, kernel_size=9, dilation=1,  stride=2,  dropout=0.05)
block2 = TemporalBlock2d(4, 8, kernel_size=9, dilation=4,  stride=1,  dropout=0.05)
block3 = TemporalBlock2d(8, 16, kernel_size=9, dilation=16,  stride=1,  dropout=0.05)

# *** IMPORTANT: Set the input length here to match what you set in prune.py ***
# If prune.py sets input to 665, use 665 here. If 833, use 833.
# Let's use 665 as per your prune.py setting.
original_input_length = 1000
print(f"--- Running traceback assuming input length: {original_input_length} ---")

# Function to calculate output length for a Conv layer with padding=0
def calculate_output_length(input_length, kernel_size, dilation, stride):
     input_length = int(input_length)
     kernel_size = int(kernel_size)
     dilation = int(dilation)
     stride = int(stride)
     # Padding is assumed 0
     padding_start = 0
     padding_end = 0
     # Correct formula for padding=0
     output_len = math.floor((input_length + padding_start + padding_end - dilation * (kernel_size - 1) - 1) / stride + 1)
     return max(0, int(output_len)) # Ensure non-negative length

# Calculate output lengths step-by-step based on the chosen input_length
output_length_L1 = calculate_output_length(original_input_length, block1.kernel_size, block1.dilation, block1.stride)
output_length_L2 = calculate_output_length(output_length_L1, block1.kernel_size, block1.dilation, 1) # block1.conv2 params
output_length_L3 = calculate_output_length(output_length_L2, block2.kernel_size, block2.dilation, block2.stride)
output_length_L4 = calculate_output_length(output_length_L3, block2.kernel_size, block2.dilation, 1) # block2.conv2 params
output_length_L5 = calculate_output_length(output_length_L4, block3.kernel_size, block3.dilation, block3.stride)
output_length_L6 = calculate_output_length(output_length_L5, block3.kernel_size, block3.dilation, 1) # block3.conv2 params

print(f"Calculated intermediate lengths (L1..L6 output): {output_length_L1}, {output_length_L2}, {output_length_L3}, {output_length_L4}, {output_length_L5}, {output_length_L6}")

# --- Check if target_output_neuron_index is valid for calculated output length ---
target_output_neuron_index = 84
if target_output_neuron_index >= output_length_L6:
    print(f"\n--- WARNING ---")
    print(f"Target output index {target_output_neuron_index} is out of bounds for the calculated final output length of {output_length_L6} (derived from input {original_input_length}).")
    print(f"This will likely result in empty relevant index lists.")
    print(f"Consider adjusting target_output_neuron_index or original_input_length.")
    print(f"Proceeding, but expect potential issues or empty prune_spec.")
    print(f"-----------------\n")
    # Optional: Adjust target index if out of bounds, e.g., clamp it
    # target_output_neuron_index = max(0, min(output_length_L6 - 1, target_output_neuron_index))
    # print(f"Adjusted target_output_neuron_index to: {target_output_neuron_index}")


# Function to get input sample influence (no change needed here)
def get_input_sample_influence(block, output_index, layer_name, input_length):
    if layer_name == 'conv1':
        conv_layer = block.conv1
        stride = block.stride
        dilation = block.dilation
        kernel_size = block.kernel_size
    elif layer_name == 'conv2':
        conv_layer = block.conv2
        stride = 1 # Always 1 for conv2
        dilation = block.dilation # Dilation same for both conv
        kernel_size = block.kernel_size
    else:
        raise ValueError("layer_name must be 'conv1' or 'conv2'")

    influenced_input_samples = []
    for kernel_index in range(kernel_size):
        # Correct calculation: output_index * stride + kernel_index * dilation
        input_sample_index = output_index * stride + kernel_index * dilation
        influenced_input_samples.append(input_sample_index)

    # Remove indices that are out of bounds using the CORRECT input_length for THIS layer
    valid_input_samples = [idx for idx in influenced_input_samples if 0 <= idx < input_length]
    return sorted(list(set(valid_input_samples)))

# --- Corrected Traceback ---

relevant_indices_per_layer = {}
non_relevant_indices_per_layer = {}

# Relevant indices in the final output tensor (Block3.conv2 output, L6)
relevant_output_indices_from_prev_layer = set([target_output_neuron_index])
relevant_indices_per_layer['block_output'] = sorted(list(relevant_output_indices_from_prev_layer))
# Non-relevant indices in the final output tensor
non_relevant_indices_per_layer['block_output'] = [i for i in range(output_length_L6) if i not in relevant_output_indices_from_prev_layer]

# Layer L6: Block 3 - conv2
current_layer_block = block3
current_layer_name = 'conv2'
input_length_to_current_layer = output_length_L5 # Input to L6 is output of L5
output_length_of_current_layer = output_length_L6

influenced_sets = [set(get_input_sample_influence(current_layer_block, idx, current_layer_name, input_length_to_current_layer))
                   for idx in relevant_output_indices_from_prev_layer]
relevant_input_indices_to_current_layer = sorted(list(reduce(set.union, influenced_sets) if influenced_sets else set()))

print(f"\n{current_layer_block.__class__.__name__} - {current_layer_name} (L6): Influenced by input samples (Output L5) at indices: {relevant_input_indices_to_current_layer}")
print(f"  Total output samples (L6): {output_length_of_current_layer}, Influencing input samples (L5 Output): {len(relevant_input_indices_to_current_layer)}")
relevant_indices_per_layer['block3_conv2'] = relevant_input_indices_to_current_layer # Relevant indices for L5 output
non_relevant_indices_per_layer['block3_conv2'] = [i for i in range(input_length_to_current_layer) if i not in relevant_input_indices_to_current_layer]
relevant_output_indices_from_prev_layer = set(relevant_input_indices_to_current_layer)

# Layer L5: Block 3 - conv1
current_layer_block = block3
current_layer_name = 'conv1'
input_length_to_current_layer = output_length_L4 # Input to L5 is output of L4
output_length_of_current_layer = output_length_L5

influenced_sets = [set(get_input_sample_influence(current_layer_block, idx, current_layer_name, input_length_to_current_layer))
                   for idx in relevant_output_indices_from_prev_layer]
relevant_input_indices_to_current_layer = sorted(list(reduce(set.union, influenced_sets) if influenced_sets else set()))

print(f"{current_layer_block.__class__.__name__} - {current_layer_name} (L5): Influenced by input samples (Output L4) at indices: {relevant_input_indices_to_current_layer}")
print(f"  Total output samples (L5): {output_length_of_current_layer}, Influencing input samples (L4 Output): {len(relevant_input_indices_to_current_layer)}")
relevant_indices_per_layer['block3_conv1'] = relevant_input_indices_to_current_layer # Relevant indices for L4 output
non_relevant_indices_per_layer['block3_conv1'] = [i for i in range(input_length_to_current_layer) if i not in relevant_input_indices_to_current_layer]
relevant_output_indices_from_prev_layer = set(relevant_input_indices_to_current_layer)

# Layer L4: Block 2 - conv2
current_layer_block = block2
current_layer_name = 'conv2'
input_length_to_current_layer = output_length_L3 # Input to L4 is output of L3
output_length_of_current_layer = output_length_L4

influenced_sets = [set(get_input_sample_influence(current_layer_block, idx, current_layer_name, input_length_to_current_layer))
                   for idx in relevant_output_indices_from_prev_layer]
relevant_input_indices_to_current_layer = sorted(list(reduce(set.union, influenced_sets) if influenced_sets else set()))

print(f"{current_layer_block.__class__.__name__} - {current_layer_name} (L4): Influenced by input samples (Output L3) at indices: {relevant_input_indices_to_current_layer}")
print(f"  Total output samples (L4): {output_length_of_current_layer}, Influencing input samples (L3 Output): {len(relevant_input_indices_to_current_layer)}")
relevant_indices_per_layer['block2_conv2'] = relevant_input_indices_to_current_layer # Relevant indices for L3 output
non_relevant_indices_per_layer['block2_conv2'] = [i for i in range(input_length_to_current_layer) if i not in relevant_input_indices_to_current_layer]
relevant_output_indices_from_prev_layer = set(relevant_input_indices_to_current_layer)

# Layer L3: Block 2 - conv1
current_layer_block = block2
current_layer_name = 'conv1'
input_length_to_current_layer = output_length_L2 # Input to L3 is output of L2
output_length_of_current_layer = output_length_L3

influenced_sets = [set(get_input_sample_influence(current_layer_block, idx, current_layer_name, input_length_to_current_layer))
                   for idx in relevant_output_indices_from_prev_layer]
relevant_input_indices_to_current_layer = sorted(list(reduce(set.union, influenced_sets) if influenced_sets else set()))

print(f"{current_layer_block.__class__.__name__} - {current_layer_name} (L3): Influenced by input samples (Output L2) at indices: {relevant_input_indices_to_current_layer}")
print(f"  Total output samples (L3): {output_length_of_current_layer}, Influencing input samples (L2 Output): {len(relevant_input_indices_to_current_layer)}")
relevant_indices_per_layer['block2_conv1'] = relevant_input_indices_to_current_layer # Relevant indices for L2 output
non_relevant_indices_per_layer['block2_conv1'] = [i for i in range(input_length_to_current_layer) if i not in relevant_input_indices_to_current_layer]
relevant_output_indices_from_prev_layer = set(relevant_input_indices_to_current_layer)

# Layer L2: Block 1 - conv2
current_layer_block = block1
current_layer_name = 'conv2'
input_length_to_current_layer = output_length_L1 # Input to L2 is output of L1
output_length_of_current_layer = output_length_L2

influenced_sets = [set(get_input_sample_influence(current_layer_block, idx, current_layer_name, input_length_to_current_layer))
                   for idx in relevant_output_indices_from_prev_layer]
relevant_input_indices_to_current_layer = sorted(list(reduce(set.union, influenced_sets) if influenced_sets else set()))

print(f"{current_layer_block.__class__.__name__} - {current_layer_name} (L2): Influenced by input samples (Output L1) at indices: {relevant_input_indices_to_current_layer}")
print(f"  Total output samples (L2): {output_length_of_current_layer}, Influencing input samples (L1 Output): {len(relevant_input_indices_to_current_layer)}")
relevant_indices_per_layer['block1_conv2'] = relevant_input_indices_to_current_layer # Relevant indices for L1 output
non_relevant_indices_per_layer['block1_conv2'] = [i for i in range(input_length_to_current_layer) if i not in relevant_input_indices_to_current_layer]
relevant_output_indices_from_prev_layer = set(relevant_input_indices_to_current_layer)

# Layer L1: Block 1 - conv1 (Original Input)
current_layer_block = block1
current_layer_name = 'conv1'
input_length_to_current_layer = original_input_length # Input to L1 is original input
output_length_of_current_layer = output_length_L1

influenced_sets = [set(get_input_sample_influence(current_layer_block, idx, current_layer_name, input_length_to_current_layer))
                   for idx in relevant_output_indices_from_prev_layer]
relevant_input_indices_to_current_layer = sorted(list(reduce(set.union, influenced_sets) if influenced_sets else set()))

print(f"{current_layer_block.__class__.__name__} - {current_layer_name} (L1): Influenced by input samples (Original Input) at indices: {relevant_input_indices_to_current_layer}")
print(f"  Total output samples (L1): {output_length_of_current_layer}, Influencing input samples (Original Input): {len(relevant_input_indices_to_current_layer)}")
relevant_indices_per_layer['block1_conv1'] = relevant_input_indices_to_current_layer # Relevant indices for Original Input
non_relevant_indices_per_layer['block1_conv1'] = [i for i in range(input_length_to_current_layer) if i not in relevant_input_indices_to_current_layer]


# --- Save results ---
print("\n--- Saving relevant and non-relevant indices to JSON ---")

# Function to convert sets to lists for JSON serialization
def make_json_serializable(data_dict):
    serializable_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, set):
            serializable_dict[key] = sorted(list(value))
        elif isinstance(value, list):
             serializable_dict[key] = sorted(list(set(value))) # Ensure unique and sorted if it's a list
        else:
             serializable_dict[key] = value # Keep non-list/set items as is
    return serializable_dict

# Serialize before saving
serializable_relevant = make_json_serializable(relevant_indices_per_layer)
serializable_non_relevant = make_json_serializable(non_relevant_indices_per_layer)

with open("relevant_indices_per_layer.json", "w") as f:
    json.dump(serializable_relevant, f, indent=2)

with open("non_relevant_indices_per_layer.json", "w") as f:
    json.dump(serializable_non_relevant, f, indent=2)

print("JSON files saved.")


# --- END OF FILE indices.py ---
















# # --- START OF FILE indices_corrected.py ---

# import torch
# import torch.nn as nn
# from brevitas import nn as qnn
# from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
# import json
# from functools import reduce
# import math

# class TemporalBlock2d(nn.Module):
#     # ... (TemporalBlock2d class definition remains the same) ...
#     def __init__(self, n_inputs, n_outputs, kernel_size, dilation, stride, dropout=0.05):
#         super(TemporalBlock2d, self).__init__()
#         self.kernel_size = kernel_size
#         self.dilation = dilation
#         self.stride = stride

#         self.conv1 = qnn.QuantConv2d(
#             n_inputs, n_outputs,
#             kernel_size=(kernel_size, 1),
#             stride=(stride, 1),
#             padding=(0, 0),  # no padding – only compute valid outputs
#             dilation=(dilation, 1),
#             weight_quant=Int8WeightPerTensorFloat,
#             input_quant=Int8ActPerTensorFloat,
#             weight_bit_width=8,
#             act_bit_width=8,
#             bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(n_outputs)
#         self.dropout1 = nn.Dropout(dropout)

#         self.conv2 = qnn.QuantConv2d(
#             n_outputs, n_outputs,
#             kernel_size=(kernel_size, 1),
#             stride=(1, 1),
#             padding=(0, 0),
#             dilation=(dilation, 1),
#             weight_quant=Int8WeightPerTensorFloat,
#             input_quant=Int8ActPerTensorFloat,
#             weight_bit_width=8,
#             act_bit_width=8,
#             bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(n_outputs)
#         self.dropout2 = nn.Dropout(dropout)

#         self.relu_after_conv = qnn.QuantReLU(return_quant_tensor=True)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.dropout1(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu_after_conv(x).value
#         x = self.dropout2(x)
#         return x

# # --- Corrected Analysis ---

# # Define blocks
# block1 = TemporalBlock2d(1, 4, kernel_size=9, dilation=1,  stride=2,  dropout=0.05)
# block2 = TemporalBlock2d(4, 8, kernel_size=9, dilation=4,  stride=1,  dropout=0.05)
# block3 = TemporalBlock2d(8, 16, kernel_size=9, dilation=16,  stride=1,  dropout=0.05)

# # Set the input length consistently
# original_input_length = 1000
# print(f"--- Running CORRECTED traceback assuming input length: {original_input_length} ---")

# # Function to calculate output length (same as before)
# def calculate_output_length(input_length, kernel_size, dilation, stride):
#      input_length = int(input_length)
#      kernel_size = int(kernel_size)
#      dilation = int(dilation)
#      stride = int(stride)
#      padding_start = 0
#      padding_end = 0
#      output_len = math.floor((input_length + padding_start + padding_end - dilation * (kernel_size - 1) - 1) / stride + 1)
#      return max(0, int(output_len))

# # Calculate output lengths step-by-step (same as before)
# output_length_L1 = calculate_output_length(original_input_length, block1.kernel_size, block1.dilation, block1.stride)
# output_length_L2 = calculate_output_length(output_length_L1, block1.kernel_size, block1.dilation, 1) # block1.conv2
# output_length_L3 = calculate_output_length(output_length_L2, block2.kernel_size, block2.dilation, block2.stride)
# output_length_L4 = calculate_output_length(output_length_L3, block2.kernel_size, block2.dilation, 1) # block2.conv2
# output_length_L5 = calculate_output_length(output_length_L4, block3.kernel_size, block3.dilation, block3.stride)
# output_length_L6 = calculate_output_length(output_length_L5, block3.kernel_size, block3.dilation, 1) # block3.conv2

# print(f"Calculated intermediate lengths (L1..L6 output): {output_length_L1}, {output_length_L2}, {output_length_L3}, {output_length_L4}, {output_length_L5}, {output_length_L6}")

# # Set target output index and check bounds (same as before)
# target_output_neuron_index = 84
# if target_output_neuron_index >= output_length_L6:
#     print(f"\n--- WARNING ---")
#     print(f"Target output index {target_output_neuron_index} is out of bounds for the calculated final output length of {output_length_L6}.")
#     # ... (rest of warning message) ...
#     # Decide how to handle: exit, clamp, etc. Let's proceed but results might be empty.
#     # target_output_neuron_index = max(0, min(output_length_L6 - 1, target_output_neuron_index))
#     # print(f"Adjusted target_output_neuron_index to: {target_output_neuron_index}")

# # Function to get input sample influence (same as before)
# def get_input_sample_influence(block, output_index, layer_name, input_length):
#     if layer_name == 'conv1':
#         stride = block.stride; dilation = block.dilation; kernel_size = block.kernel_size
#     elif layer_name == 'conv2':
#         stride = 1; dilation = block.dilation; kernel_size = block.kernel_size
#     else: raise ValueError("layer_name must be 'conv1' or 'conv2'")
#     influenced_input_samples = [output_index * stride + k * dilation for k in range(kernel_size)]
#     return {idx for idx in influenced_input_samples if 0 <= idx < input_length} # Return a set directly

# # --- Traceback using UNION ---

# # Map to store the full set of necessary indices FROM the output tensor of each stage
# # Keys will represent the tensor (e.g., 'output_L5', 'output_L4', ...)
# necessary_indices_map = {}

# # Start with the target indices needed FROM the final output (L6)
# needed_indices_from_current_stage_output = {target_output_neuron_index}
# necessary_indices_map['output_L6'] = needed_indices_from_current_stage_output

# # --- Layer L6 (Block 3, conv2) -> Needs input from L5 ---
# current_block = block3
# current_layer = 'conv2'
# input_len = output_length_L5 # Input to L6 is output of L5
# output_len = output_length_L6

# # Find all inputs from L5 needed for *any* of the required outputs in L6
# influenced_sets = [get_input_sample_influence(current_block, idx, current_layer, input_len)
#                    for idx in needed_indices_from_current_stage_output if 0 <= idx < output_len] # Check output index bounds
# needed_indices_from_current_stage_output = reduce(set.union, influenced_sets) if influenced_sets else set()
# necessary_indices_map['output_L5'] = needed_indices_from_current_stage_output # Store indices needed FROM L5's output

# # --- Layer L5 (Block 3, conv1) -> Needs input from L4 ---
# current_block = block3
# current_layer = 'conv1'
# input_len = output_length_L4 # Input to L5 is output of L4
# output_len = output_length_L5

# influenced_sets = [get_input_sample_influence(current_block, idx, current_layer, input_len)
#                    for idx in needed_indices_from_current_stage_output if 0 <= idx < output_len]
# needed_indices_from_current_stage_output = reduce(set.union, influenced_sets) if influenced_sets else set()
# necessary_indices_map['output_L4'] = needed_indices_from_current_stage_output # Store indices needed FROM L4's output

# # --- Layer L4 (Block 2, conv2) -> Needs input from L3 ---
# current_block = block2
# current_layer = 'conv2'
# input_len = output_length_L3 # Input to L4 is output of L3
# output_len = output_length_L4

# influenced_sets = [get_input_sample_influence(current_block, idx, current_layer, input_len)
#                    for idx in needed_indices_from_current_stage_output if 0 <= idx < output_len]
# needed_indices_from_current_stage_output = reduce(set.union, influenced_sets) if influenced_sets else set()
# necessary_indices_map['output_L3'] = needed_indices_from_current_stage_output # Store indices needed FROM L3's output

# # --- Layer L3 (Block 2, conv1) -> Needs input from L2 ---
# current_block = block2
# current_layer = 'conv1'
# input_len = output_length_L2 # Input to L3 is output of L2
# output_len = output_length_L3

# influenced_sets = [get_input_sample_influence(current_block, idx, current_layer, input_len)
#                    for idx in needed_indices_from_current_stage_output if 0 <= idx < output_len]
# needed_indices_from_current_stage_output = reduce(set.union, influenced_sets) if influenced_sets else set()
# necessary_indices_map['output_L2'] = needed_indices_from_current_stage_output # Store indices needed FROM L2's output

# # --- Layer L2 (Block 1, conv2) -> Needs input from L1 ---
# current_block = block1
# current_layer = 'conv2'
# input_len = output_length_L1 # Input to L2 is output of L1
# output_len = output_length_L2

# influenced_sets = [get_input_sample_influence(current_block, idx, current_layer, input_len)
#                    for idx in needed_indices_from_current_stage_output if 0 <= idx < output_len]
# needed_indices_from_current_stage_output = reduce(set.union, influenced_sets) if influenced_sets else set()
# necessary_indices_map['output_L1'] = needed_indices_from_current_stage_output # Store indices needed FROM L1's output

# # --- Layer L1 (Block 1, conv1) -> Needs input from Original Input ---
# current_block = block1
# current_layer = 'conv1'
# input_len = original_input_length # Input to L1 is original input
# output_len = output_length_L1

# influenced_sets = [get_input_sample_influence(current_block, idx, current_layer, input_len)
#                    for idx in needed_indices_from_current_stage_output if 0 <= idx < output_len]
# needed_indices_from_current_stage_output = reduce(set.union, influenced_sets) if influenced_sets else set()
# necessary_indices_map['original_input'] = needed_indices_from_current_stage_output # Store indices needed FROM original input

# # --- Reorganize and Save the Corrected Map ---
# # Create the mapping from Node Name -> Indices FROM that node's output
# # based on the necessary_indices_map calculated above.

# # This map is now suitable for defining slices *after* the named node.
# node_to_required_output_indices = {
#     "global_in"                 : sorted(list(necessary_indices_map.get('original_input', set()))),
#     "temporal_blocks.0.conv1"   : sorted(list(necessary_indices_map.get('output_L1', set()))),
#     "temporal_blocks.0.conv2"   : sorted(list(necessary_indices_map.get('output_L2', set()))),
#     "temporal_blocks.1.conv1"   : sorted(list(necessary_indices_map.get('output_L3', set()))),
#     "temporal_blocks.1.conv2"   : sorted(list(necessary_indices_map.get('output_L4', set()))),
#     "temporal_blocks.2.conv1"   : sorted(list(necessary_indices_map.get('output_L5', set()))),
#     "temporal_blocks.2.conv2"   : sorted(list(necessary_indices_map.get('output_L6', set()))), # Should contain [84] if valid
# }

# print("\n--- Corrected Necessary Indices Map (Union Method) ---")
# # Use the previous print function or a similar one
# def print_indices_map_summary(indices_map):
#     print("\n📋 Indices Per Node (Required FROM this node's output):\n")
#     max_key_len = max(len(k) for k in indices_map.keys()) if indices_map else 0
#     for node, indices in indices_map.items():
#         key_str = f"🔹 {node}:".ljust(max_key_len + 4)
#         if not isinstance(indices, list): # Should always be list now
#             print(f"{key_str}❓ Invalid format")
#             continue
#         count = len(indices)
#         if count == 0:
#              print(f"{key_str} ❌ 0 indices")
#              continue
#         first = indices[:5]
#         last = indices[-5:] if count > 5 else []
#         preview = f"{first} ... {last}" if last else f"{first}"
#         print(f"{key_str} {count} required indices → {preview}")

# print_indices_map_summary(node_to_required_output_indices)


# # Save this corrected map
# output_filename = "corrected_relevant_indices_per_model_layer.json" # New name
# try:
#     with open(output_filename, "w") as f:
#         json.dump(node_to_required_output_indices, f, indent=2)
#     print(f"\n✅ Successfully saved **corrected** node-to-indices map to '{output_filename}'")
# except Exception as e:
#     print(f"\n❌ Error saving JSON file '{output_filename}': {e}")


# # --- END OF FILE indices_corrected.py ---