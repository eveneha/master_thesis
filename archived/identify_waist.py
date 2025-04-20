import pandas as pd

# Define your TCN layers in order
layers = [
    ("block1.conv1", 9, 1, 2),
    ("block1.conv2", 9, 1, 1),
    ("block2.conv1", 9, 4, 1),
    ("block2.conv2", 9, 4, 1),
    ("block3.conv1", 9, 16, 1),
    ("block3.conv2", 9, 16, 1),
]

output_index = 84
desired_input_center = 500
input_length = 1000  # total raw input samples

def forward_trace_input_center(output_idx, layers):
    idx = output_idx
    for _, k, d, s in reversed(layers):
        eff_k = (k - 1) * d + 1
        idx = idx * s + eff_k // 2
    return idx

def backtrace_indices_for_layer(out_indices, kernel, dilation, stride):
    in_indices = set()
    for out_idx in out_indices:
        center = out_idx * stride
        for i in range(kernel):
            idx = center + (i - kernel // 2) * dilation
            in_indices.add(idx)
    return sorted(in_indices)

# 1) Compute where output[84] maps to in the raw input
mapped_center = forward_trace_input_center(output_index, layers)
shift = desired_input_center - mapped_center

# 2) Back‐trace for each layer and apply shift & clamp to [0, input_length)
current_indices = {output_index}
results = []

for name, kernel, dilation, stride in reversed(layers):
    raw = backtrace_indices_for_layer(current_indices, kernel, dilation, stride)
    aligned = [i + shift for i in raw]
    aligned = [i for i in aligned if 0 <= i < input_length]
    
    results.append({
        "Layer": name,
        "Raw Count": len(raw),
        "Raw Range": f"{min(raw)}…{max(raw)}",
        "Aligned Count": len(aligned),
        "Aligned Range": f"{min(aligned)}…{max(aligned)}"
    })
    
    current_indices = set(raw)

# Flip back to original layer order
results.reverse()

import pandas as pd

# Define your TCN layers in order
layers = [
    ("block1.conv1", 9, 1, 2),
    ("block1.conv2", 9, 1, 1),
    ("block2.conv1", 9, 4, 1),
    ("block2.conv2", 9, 4, 1),
    ("block3.conv1", 9, 16, 1),
    ("block3.conv2", 9, 16, 1),
]

output_index = 84
desired_input_center = 500
input_length = 1000  # total raw input samples

def forward_trace_input_center(output_idx, layers):
    idx = output_idx
    for _, k, d, s in reversed(layers):
        eff_k = (k - 1) * d + 1
        idx = idx * s + eff_k // 2
    return idx

def backtrace_indices_for_layer(out_indices, kernel, dilation, stride):
    in_indices = set()
    for out_idx in out_indices:
        center = out_idx * stride
        for i in range(kernel):
            idx = center + (i - kernel // 2) * dilation
            in_indices.add(idx)
    return sorted(in_indices)

# 1) Map output[84] to raw input index
mapped_center = forward_trace_input_center(output_index, layers)
print(f"Output index {output_index} maps to raw input index {mapped_center}")

# 2) Compute shift to align that to desired input center (500)
shift = desired_input_center - mapped_center
print(f"Applying shift of {shift} to back-traced indices\n")

# 3) Backtrace and print for each layer
current_indices = {output_index}
for name, k, d, s in reversed(layers):
    raw = backtrace_indices_for_layer(current_indices, k, d, s)
    aligned = [i + shift for i in raw]
    # clamp
    aligned = [i for i in aligned if 0 <= i < input_length]
    print(f"{name}:")
    print(f"  Raw  indices count {len(raw)}, range {min(raw)}…{max(raw)}")
    print(f"  Aligned indices count {len(aligned)}, range {min(aligned)}…{max(aligned)}\n")
    current_indices = set(raw)

# When you run this script, you'll see exactly the raw and aligned ranges for each layer.

