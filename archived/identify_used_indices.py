import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from brevitas import nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat

class TemporalBlock2d(nn.Module):
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

block1 = TemporalBlock2d(1, 4, kernel_size=9, dilation=1,  stride=2,  dropout=0.05)
block2 = TemporalBlock2d(4, 8, kernel_size=9, dilation=4,  stride=1,  dropout=0.05)
block3 = TemporalBlock2d(8, 16, kernel_size=9, dilation=16,  stride=1,  dropout=0.05)

input_tensor = torch.randn(1, 1, 1000, 1) # [N, C_in, L_in, 1] -  assuming temporal dimension is L

output_block1 = block1(input_tensor)
output_block2 = block2(output_block1)
output_block3 = block3(output_block2)


target_output_neuron_index = 84 # We are interested in output neuron number 84 in the last conv of block3

# --- Analysis ---

def get_input_sample_influence(block, output_index, layer_name, input_length):
    """
    Determines the input sample indices that influence a given output index
    for a convolutional layer in a TemporalBlock2d.

    Args:
        block: The TemporalBlock2d instance.
        output_index: The index of the output neuron we are interested in (int).
        layer_name: 'conv1' or 'conv2' to specify the convolutional layer.
        input_length: The length of the input tensor to this layer.

    Returns:
        A list of input sample indices that influence the output neuron.
    """
    if layer_name == 'conv1':
        conv_layer = block.conv1
        stride = block.stride
        dilation = block.dilation
        kernel_size = block.kernel_size
    elif layer_name == 'conv2':
        conv_layer = block.conv2
        stride = 1 # Always 1 for conv2 in this setup
        dilation = block.dilation # Dilation same for both conv layers in block
        kernel_size = block.kernel_size
    else:
        raise ValueError("layer_name must be 'conv1' or 'conv2'")


    influenced_input_samples = []
    for kernel_index in range(kernel_size):
        input_sample_index = output_index * stride + kernel_index * dilation
        influenced_input_samples.append(input_sample_index)

    # Remove indices that are out of bounds
    valid_input_samples = [idx for idx in influenced_input_samples if 0 <= idx < input_length] # Check against input_length now
    return sorted(list(set(valid_input_samples))) # Remove duplicates and sort


# --- Traceback for block3's conv2 output 84 ---

current_output_index = target_output_neuron_index
relevant_input_indices = [current_output_index] # Start with the target output neuron

print(f"Target output neuron in block3.conv2: {target_output_neuron_index}")

# Block 3 - conv2
input_length_block3_conv2 = output_block2.shape[2] # Input length to block3.conv2 is output length of block2
output_length_block3_conv2 = output_block3.shape[2]
block3_conv2_influenced_indices = []
for output_index in relevant_input_indices: # In first iteration, this is just [84]
    indices = get_input_sample_influence(block3, output_index, 'conv2', input_length_block3_conv2) # Input length to conv2
    block3_conv2_influenced_indices.extend(indices)
block3_conv2_influenced_indices = sorted(list(set(block3_conv2_influenced_indices)))
print(f"\nBlock3 - conv2: Output neuron {target_output_neuron_index} is influenced by input samples from block3 - conv1 at indices: {block3_conv2_influenced_indices}")
print(f"Block3 - conv2: Total output samples: {output_length_block3_conv2}, Total influencing input samples: {len(block3_conv2_influenced_indices)}")
relevant_input_indices = block3_conv2_influenced_indices # Update for next layer


# Block 3 - conv1
input_length_block3_conv1 = output_block1.shape[2] # Input length to block3.conv1 is output length of block1
output_length_block3_conv1 = output_block2.shape[2]
block3_conv1_influenced_indices = []
for output_index in relevant_input_indices:
    indices = get_input_sample_influence(block3, output_index, 'conv1', input_length_block3_conv1) # Input length to conv1
    block3_conv1_influenced_indices.extend(indices)
block3_conv1_influenced_indices = sorted(list(set(block3_conv1_influenced_indices)))
print(f"Block3 - conv1: Influenced by input samples from block2 at indices: {block3_conv1_influenced_indices}")
print(f"Block3 - conv1: Total output samples: {output_length_block3_conv1}, Total influencing input samples: {len(block3_conv1_influenced_indices)}")
relevant_input_indices = block3_conv1_influenced_indices


# Block 2 - conv2
input_length_block2_conv2 = output_block1.shape[2] # Input length to block2.conv2 is output length of block1
output_length_block2_conv2 = output_block2.shape[2]
block2_conv2_influenced_indices = []
for output_index in relevant_input_indices:
    indices = get_input_sample_influence(block2, output_index, 'conv2', input_length_block2_conv2) # Input length to conv2
    block2_conv2_influenced_indices.extend(indices)
block2_conv2_influenced_indices = sorted(list(set(block2_conv2_influenced_indices)))
print(f"\nBlock2 - conv2: Influenced by input samples from block2 - conv1 at indices: {block2_conv2_influenced_indices}")
print(f"Block2 - conv2: Total output samples: {output_length_block2_conv2}, Total influencing input samples: {len(block2_conv2_influenced_indices)}")
relevant_input_indices = block2_conv2_influenced_indices

# Block 2 - conv1
input_length_block2_conv1 = input_tensor.shape[2] # Input length to block2.conv1 is input tensor length
output_length_block2_conv1 = output_block1.shape[2]
block2_conv1_influenced_indices = []
for output_index in relevant_input_indices:
    indices = get_input_sample_influence(block2, output_index, 'conv1', input_length_block2_conv1) # Input length to conv1
    block2_conv1_influenced_indices.extend(indices)
block2_conv1_influenced_indices = sorted(list(set(block2_conv1_influenced_indices)))
print(f"Block2 - conv1: Influenced by input samples from block1 at indices: {block2_conv1_influenced_indices}")
print(f"Block2 - conv1: Total output samples: {output_length_block2_conv1}, Total influencing input samples: {len(block2_conv1_influenced_indices)}")
relevant_input_indices = block2_conv1_influenced_indices


# Block 1 - conv2
input_length_block1_conv2 = output_block1.shape[2] # Input length to block1.conv2 is output length of block1.conv1
output_length_block1_conv2 = output_block1.shape[2] #Output length is same as input length for conv2 in block1
block1_conv2_influenced_indices = []
for output_index in relevant_input_indices:
    indices = get_input_sample_influence(block1, output_index, 'conv2', input_length_block1_conv2) # Input length to conv2
    block1_conv2_influenced_indices.extend(indices)
block1_conv2_influenced_indices = sorted(list(set(block1_conv2_influenced_indices)))
print(f"\nBlock1 - conv2: Influenced by input samples from block1 - conv1 at indices: {block1_conv2_influenced_indices}")
print(f"Block1 - conv2: Total output samples: {output_length_block1_conv2}, Total influencing input samples: {len(block1_conv2_influenced_indices)}")
relevant_input_indices = block1_conv2_influenced_indices


# Block 1 - conv1 (Original Input)
input_length_block1_conv1 = input_tensor.shape[2] # Input length to block1.conv1 is input tensor length
output_length_block1_conv1 = output_block1.shape[2]
block1_conv1_influenced_indices = []
for output_index in relevant_input_indices:
    indices = get_input_sample_influence(block1, output_index, 'conv1', input_length_block1_conv1) # Input length to conv1
    block1_conv1_influenced_indices.extend(indices)
block1_conv1_influenced_indices = sorted(list(set(block1_conv1_influenced_indices)))
print(f"Block1 - conv1: Influenced by input samples from the original input tensor at indices: {block1_conv1_influenced_indices}")
print(f"Block1 - conv1: Total output samples: {output_length_block1_conv1}, Total influencing input samples: {len(block1_conv1_influenced_indices)}")
relevant_input_indices = block1_conv1_influenced_indices