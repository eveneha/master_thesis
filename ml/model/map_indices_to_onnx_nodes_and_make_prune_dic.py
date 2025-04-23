import json

with open("relevant_indices_per_layer.json", "r") as f:
    relevant_indices_per_layer = json.load(f)

with open("non_relevant_indices_per_layer.json", "r") as f:
    non_relevant_indices_per_layer = json.load(f)
    
def print_relevant_indices_map(indices_map):
    print("\n📋 Indices Per Layer:\n")
    for node, indices in indices_map.items():
        if not indices:
            print(f"🔹 {node}: ❌ No indices")
            continue
        try :
            count = len(indices)
        except TypeError:
            count = 1
            indices = [indices]    
        first = indices[:5]
        last = indices[-5:] if count > 5 else []
        preview = f"{first} ... {last}" if last else f"{first}"

        print(f"🔹 {node}: {count} indices → {preview}")


def create_node_index_map(base_map, map_name="relevant_indices_per_node"):
    return {
        #"global_in"            : base_map["block1_conv1"],
        "Quant_0"              : base_map["block1_conv1"],
        "Quant_1"              : base_map["block1_conv1"],
        
        "Conv_0"               : base_map["block1_conv2"],
        "Mul_0"                : base_map["block1_conv2"],
        "BatchNormalization_0" : base_map["block1_conv2"],
        "Quant_2"              : base_map["block1_conv2"],        
        
        "Conv_1"               : base_map["block1_conv2"][4:-4], ## this one
 
        "Mul_1"                : base_map["block2_conv1"],
        "BatchNormalization_1" : base_map["block2_conv1"],
        "Relu_0"               : base_map["block2_conv1"],
        "Quant_3"              : base_map["block2_conv1"],
        "Quant_4"              : base_map["block2_conv1"],
        
        "Conv_2"               : base_map["block2_conv2"],
        "Mul_2"                : base_map["block2_conv2"],
        "BatchNormalization_2" : base_map["block2_conv2"],
        "Quant_5"              : base_map["block2_conv2"],
        
        "Conv_3"               : base_map["block2_conv2"][4:-4], ## ti
        
        "Mul_3"                : base_map["block3_conv1"],
        "BatchNormalization_3" : base_map["block3_conv1"],
        "Relu_1"               : base_map["block3_conv1"],
        "Quant_6"              : base_map["block3_conv1"],
        "Quant_7"              : base_map["block3_conv1"],
        
        "Conv_4"               : base_map["block3_conv2"],
        "Mul_4"                : base_map["block3_conv2"],
        "BatchNormalization_4" : base_map["block3_conv2"],
        "Quant_8"              : base_map["block3_conv2"],
        
        "Conv_5"               : base_map["block_output"],
        "Mul_5"                : base_map["block_output"],
        "BatchNormalization_5" : base_map['block_output'],
        "Relu_2"               : base_map['block_output'],
        "Quant_9"              : base_map['block_output'],
        "Quant_10"             : base_map['block_output'],
    }
    

    
print(f"Original block1_conv2 len: {len(relevant_indices_per_layer['block1_conv2'])}")
print(f"Sliced block1_conv2[4:-4] len: {len(relevant_indices_per_layer['block1_conv2'][4:-4])}")


relevant_indices_per_node = create_node_index_map(relevant_indices_per_layer)
non_relevant_indices_per_node = create_node_index_map(non_relevant_indices_per_layer)

print_relevant_indices_map(relevant_indices_per_node)


def create_prune_spec(non_relevant_indices_per_node):
    prune_spec = {}
    for node_name, indices in non_relevant_indices_per_node.items():
        prune_spec[f"{node_name}_out0"] = {2: set(indices)}  # assuming axis=2 is correct
    return prune_spec

def save_prune_spec(prune_spec, path="prune_spec.json"):
    # Convert sets to lists for JSON compatibility
    serializable_spec = {
        tensor_name: {axis: list(indices) for axis, indices in axis_map.items()}
        for tensor_name, axis_map in prune_spec.items()
    }
    with open(path, "w") as f:
        json.dump(serializable_spec, f, indent=2)
        
        
prune_spec = create_prune_spec(non_relevant_indices_per_node)
save_prune_spec(prune_spec, path="prune_spec.json")





# # --- START OF FILE map_indices_to_onnx_nodes_and_make_prune_dic.py (Gather Map Version) ---
# import json

# # Load indices calculated based on target input size (e.g., 665)
# try:
#     with open("relevant_indices_per_layer.json", "r") as f:
#         relevant_indices_per_layer = json.load(f)
# except FileNotFoundError:
#     print("Error: relevant_indices_per_layer.json not found. Run corrected indices.py first.")
#     exit()
# except json.JSONDecodeError:
#     print("Error: Could not decode relevant_indices_per_layer.json.")
#     exit()

# # (Keep print_indices_map function if desired for checking)
# def print_indices_map(indices_map):
#     print("\n📋 Indices Per Mapped Tensor:\n")
#     for node, indices in indices_map.items():
#         if not indices and isinstance(indices, list): print(f"🔹 {node}: ❌ No indices"); continue
#         try: count = len(indices)
#         except TypeError: count = 1; indices = [indices]
#         except Exception as e: print(f"Error processing indices for {node}: {e}"); continue
#         first = indices[:5]; last = indices[-5:] if count > 5 else []; preview = f"{first} ... {last}" if last else f"{first}"
#         print(f"🔹 {node}: {count} indices → {preview}")

# # --- Create the map for GATHER NODES ---
# # Maps specific ONNX OUTPUT TENSOR names to RELEVANT indices from base_map
# # *** IMPORTANT: Verify these ONNX tensor names using Netron on model_cleaned_input_set.onnx ***
# def create_gather_map(base_map_relevant):
#      mapping = {}
#      # Insert after Conv_1 (output of Block2.conv1)
#      # Check Netron for the actual output tensor name of the Conv node named "Conv_1"
#      mapping["Conv_1_out0"] = base_map_relevant.get("block2_conv1", [])

#      # Insert after Conv_3 (output of Block3.conv1)
#      # Check Netron for the actual output tensor name of the Conv node named "Conv_3"
#      mapping["Conv_3_out0"] = base_map_relevant.get("block3_conv1", [])

#      # Insert after the last node of Block 3 before the FC/Slice
#      # Based on your previous mapping, this seems to be Quant_8_out0
#      # We need to gather the SINGLE relevant index here (index 84 relative to original Block 3 output)
#      # The indices in base_map["block_output"] should contain just [84]
#      #mapping["Quant_8_out0"] = base_map_relevant.get("block_output", [])

#      # Remove entries with empty lists
#      mapping = {k: v for k, v in mapping.items() if v}
#      return mapping

# gather_map = create_gather_map(relevant_indices_per_layer)

# print("--- Gather Map (Relevant Indices for InsertTemporalGather) ---")
# print_indices_map(gather_map)

# # Function to save JSON
# def save_json(data, path):
#     serializable_data = {}
#     for key, value in data.items():
#          # Ensure values are lists of integers for JSON
#          serializable_data[key] = sorted(list(set(int(i) for i in value)))
#     with open(path, "w") as f:
#         json.dump(serializable_data, f, indent=2)

# # Save gather_map.json for InsertTemporalGather
# save_json(gather_map, path="gather_map.json")
# print("\ngather_map.json file created (for InsertTemporalGather).")

# # # --- END OF FILE ---

# # --- START OF FILE map_indices...py (Generates BOTH spec files) ---
# import json

# # Load indices calculated based on 1000-sample input
# try:
#     with open("relevant_indices_per_layer.json", "r") as f:
#         relevant_indices_per_layer = json.load(f)
#     with open("non_relevant_indices_per_layer.json", "r") as f:
#         non_relevant_indices_per_layer = json.load(f)
# except Exception as e: print(f"Error loading index files: {e}"); exit()

# # (Keep print_indices_map function)
# def print_indices_map(indices_map, title):
#     print(f"\n📋 {title}:\n")
#     for node, indices in indices_map.items():
#         # ... (rest of print function as before) ...
#         if not indices and isinstance(indices, list): print(f"🔹 {node}: ❌ No indices"); continue
#         try: count = len(indices)
#         except TypeError: count = 1; indices = [indices]
#         except Exception as e: print(f"Error processing indices for {node}: {e}"); continue
#         first = indices[:5]; last = indices[-5:] if count > 5 else []; preview = f"{first} ... {last}" if last else f"{first}"
#         print(f"🔹 {node}: {count} indices → {preview}")

# # --- Function to create the map for GATHER NODES (relevant, specific Conv outputs ONLY) ---
# def create_gather_map(base_map_relevant):
#      # Maps specific ONNX CONV OUTPUT TENSOR names to RELEVANT indices
#      # *** VERIFY THESE TENSOR NAMES WITH NETRON ***
#      mapping = {}
#      mapping["Conv_1"] = base_map_relevant.get("block2_conv1", []) # Output of Block2.conv1
#      mapping["Conv_3"] = base_map_relevant.get("block3_conv1", []) # Output of Block3.conv1
#      # Add gather for final block output before FC/Slice
#      #mapping["Quant_8_out0"] = base_map_relevant.get("block_output", [])

#      return {k: v for k, v in mapping.items() if v} # Remove empty lists


# # Create the maps
# gather_map = create_gather_map(relevant_indices_per_layer)
# print_indices_map(gather_map, "Gather Map (Relevant for InsertConvGatherNodes)")

# # Function to save JSON
# def save_json(data, path):
#     # Convert sets to lists for JSON compatibility if needed within data structure
#     def default_serializer(obj):
#         if isinstance(obj, set): return sorted(list(obj))
#         # Handle potential lists within the map values directly
#         if isinstance(obj, list): return sorted(list(set(int(i) for i in obj if isinstance(i, (int, str)) and str(i).isdigit())))
#         raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

#     serializable_data = {}
#     for key, value in data.items():
#          if isinstance(value, dict): # For prune_spec format {tensor: {axis: indices}}
#               serializable_data[key] = {int(k): default_serializer(v) for k,v in value.items()}
#          else: # For gather_map format {tensor: indices}
#               serializable_data[key] = default_serializer(value)

#     with open(path, "w") as f:
#         json.dump(serializable_data, f, indent=2)


# # Save gather_map.json for InsertConvGatherNodes
# save_json(gather_map, path="gather_map.json")
# print("gather_map.json file created (for InsertConvGatherNodes).")
# # --- END OF FILE ---