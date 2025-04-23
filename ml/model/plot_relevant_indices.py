import json
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce # Might be needed if processing sets

def plot_relevant_indices(relevant_indices_map, input_length=None, title="Relevant Temporal Samples per Layer"):
    """
    Plots relevant temporal sample indices for each layer as a line of dots.

    Args:
        relevant_indices_map (dict): Dictionary where keys are layer names
                                     and values are lists of relevant indices.
        input_length (int, optional): The original input length, used to set
                                      the x-axis limit appropriately.
                                      Defaults to None (calculates from data).
        title (str, optional): The title for the plot.
    """

    if not relevant_indices_map:
        print("No indices data to plot.")
        return

    # --- Define the desired plotting order (bottom to top) ---
    # Adjust these keys exactly as they appear in your JSON file
    # This order assumes the traceback calculation order
    plot_order = [
         'block1_conv1', # Represents relevant indices in Original Input
         'block1_conv2', # Represents relevant indices in Output of Block1.conv1
         'block2_conv1', # Represents relevant indices in Output of Block1.conv2
         'block2_conv2', # Represents relevant indices in Output of Block2.conv1
         'block3_conv1', # Represents relevant indices in Output of Block2.conv2
         'block3_conv2', # Represents relevant indices in Output of Block3.conv1
         'block_output'  # Represents relevant index in Output of Block3.conv2 (before FC/Slice)
    ]
    # Filter keys to only those present in the input map, maintaining order
    layer_keys_ordered = [k for k in plot_order if k in relevant_indices_map]

    if not layer_keys_ordered:
        print("None of the specified layer keys found in the data.")
        return

    num_layers = len(layer_keys_ordered)
    # Adjust figure size dynamically based on number of layers
    fig_height = max(4, num_layers * 0.6 + 1) # Ensure minimum height
    fig, ax = plt.subplots(figsize=(15, fig_height))

    max_x = 0
    all_y_coords = list(range(num_layers))
    layer_labels = []

    print("--- Plotting Relevant Indices ---")
    for i, layer_name in enumerate(layer_keys_ordered):
        indices = relevant_indices_map.get(layer_name, [])
        y_coord = i # Assign y-coordinate (0 for input, increasing upwards)

        if indices:
             # Ensure indices are numbers and handle potential single value case
             try:
                 indices_np = np.array(indices, dtype=int)
                 if indices_np.ndim == 0: # Handle single index
                     indices_np = np.array([indices_np.item()])
             except ValueError:
                  print(f"Warning: Could not convert indices for layer '{layer_name}' to integers. Skipping.")
                  layer_labels.append(f"{layer_name} (Error)")
                  continue


             print(f"Layer: {layer_name:<15} | Relevant Indices: {len(indices_np):<5} | Y-Coord: {y_coord}")
             ax.scatter(indices_np, [y_coord] * len(indices_np), marker='.', s=15, label=layer_name, alpha=0.8)

             current_max = np.max(indices_np) if len(indices_np) > 0 else 0
             max_x = max(max_x, current_max)
             layer_labels.append(f"{layer_name} ({len(indices_np)})") # Add count to label

        else:
             print(f"Layer: {layer_name:<15} | Relevant Indices: 0      | Y-Coord: {y_coord}")
             layer_labels.append(f"{layer_name} (0)") # Indicate zero count

    # --- Configure Plot ---
    ax.set_yticks(all_y_coords)
    ax.set_yticklabels(layer_labels)
    ax.set_xlabel("Temporal Sample Index")
    ax.set_ylabel("Network Layer/Stage (Output Indices)")
    ax.set_title(title)

    # Determine x-limit
    if input_length is not None:
         plot_xlim_max = input_length
    elif max_x > 0:
         plot_xlim_max = max_x * 1.05 # Add some padding
    else:
         plot_xlim_max = 100 # Default if no data

    ax.set_xlim(-5, plot_xlim_max) # Start slightly before 0
    ax.set_ylim(-0.5, num_layers - 0.5) # Adjust ylim for better spacing

    ax.grid(True, axis='x', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.grid(True, axis='y', linestyle='-', linewidth=0.5, alpha=0.5)
    # Optional: Add legend if needed, might be cluttered
    # ax.legend(markerscale=3, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

# --- How to Use ---

# 1. Make sure you have run the CORRECTED `indices.py` first
#    (using the desired input length, e.g., 665)
#    and it has generated "relevant_indices_per_layer.json".

# 2. Run this plotting code:
try:
    with open("relevant_indices_per_layer.json", "r") as f:
        relevant_indices_data = json.load(f)

    # Provide the input length used when generating the indices file
    # This ensures the x-axis scale is correct relative to the original input
    input_len_for_indices = 665 # Or 1000, depending on indices.py run

    plot_relevant_indices(
        relevant_indices_data,
        input_length=input_len_for_indices,
        title=f"Relevant Temporal Samples (Input Length = {input_len_for_indices})"
    )

except FileNotFoundError:
    print("Error: relevant_indices_per_layer.json not found. Please run the corrected indices.py script first.")
except json.JSONDecodeError:
    print("Error: Could not decode JSON from relevant_indices_per_layer.json. Check the file format.")
except Exception as e:
    print(f"An error occurred during plotting: {e}")