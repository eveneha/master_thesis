import torch
import numpy as np
import os
from collections import Counter

# --- Paths ---
save_dir = "/home/eveneiha/finn/workspace/ml/data/"
pt_path = os.path.join(save_dir, "test_quantized.pt")
npy_input_path = os.path.join(save_dir, "input.npy")
npy_label_path = os.path.join(save_dir, "labels.npy")

# --- Load Original Data ---
test_data = torch.load(os.path.join(save_dir, "test.pt"))
test_inputs = test_data["inputs"]
test_labels = test_data["labels"]

# --- Print label stats ---
label_counter = Counter(label.item() for label in test_labels)
for label, count in sorted(label_counter.items()):
    print(f"Label {label}: {count} samples")

print("Original dtype:", test_inputs.dtype)
print("Shape:", test_inputs.shape)

# --- Quantize Inputs ---
# Original: (N, 1, 1000) → Desired: (N, 1000, 1, 1)
test_inputs = test_inputs.permute(0, 2, 3, 1)

### --- YOU ARE ALLOWED TO CHANGE CODE BELOW THIS --- ###


inp_scale = 0.01801544427871704                          # <<< UPDATE THIS FOR EVERY NEW TRAINING SESH                                        
print(f"Using input scale from model: {inp_scale:.6f}")

# 1) Quantize entire test set with _that_ scale
quantized_inputs = (test_inputs.to(torch.float32) / inp_scale).round().clamp(-127, 127).to(torch.int8)

# 2) Quantize your single‐sample .npy exactly the same way
finn_sample = np.load(os.path.join(save_dir, "finn_sample.npy")).astype(np.float32)
quantized_finn_sample = np.clip(np.round(finn_sample / inp_scale), -127, 127).astype(np.int8)


### --- YOU ARE ALLOWED TO CHANGE CODE ABOVE THIS --- ###
np.save(os.path.join(save_dir, "finn_sample_quantized.npy"), quantized_finn_sample)

#torch.save(torch.from_numpy(quantized_finn_sample), os.path.join(save_dir, "finn_sample_quantized.pt"))

print("Final quantized input shape:", quantized_inputs.shape)

# --- Save .pt and .npy formats ---
torch.save((quantized_inputs, test_labels), pt_path)
np.save(npy_input_path, quantized_inputs.numpy())
np.save(npy_label_path, test_labels.numpy())

print("✅ Saved:")
print(f"  • {pt_path}")
print(f"  • {npy_input_path}")
print(f"  • {npy_label_path}")
print(f"  • {os.path.join(save_dir, 'finn_sample_quantized.npy')}")

