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
        #"Quant_0"               : base_map["block1_conv1"],
        "Quant_1"              : base_map["block1_conv1"],
        
        "Conv_0"               : base_map["block1_conv2"],
        "Mul_0"                : base_map["block1_conv2"],
        "BatchNormalization_0" : base_map["block1_conv2"],
        "Quant_2"              : base_map["block1_conv2"],
        
        "Conv_1"               : base_map["block2_conv1"],
        "Mul_1"                : base_map["block2_conv1"],
        "BatchNormalization_1" : base_map["block2_conv1"],
        "Relu_0"               : base_map["block2_conv1"],
        "Quant_3"              : base_map["block2_conv1"],
        "Quant_4"              : base_map["block2_conv1"],
        
        "Conv_2"               : base_map["block2_conv2"],
        "Mul_2"                : base_map["block2_conv2"],
        "BatchNormalization_2" : base_map["block2_conv2"],
        "Quant_5"              : base_map["block2_conv2"],
        
        "Conv_3"               : base_map["block3_conv1"],
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