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
        "global_in"                                 : base_map["block1_conv1"],
                
        "temporal_blocks.0.conv1"                   : base_map["block1_conv2"],
        #"BatchNorm1"            : base_map["block1_conv2"],
        
        "temporal_blocks.0.conv2"                 : base_map["block2_conv1"], ## this one
 
        #"BatchNorm2"            : base_map["block2_conv1"],
        #"Relu1"                 : base_map["block2_conv1"],
        
        "temporal_blocks.1.conv1"                 : base_map["block2_conv2"],
        #"BatchNorm3"            : base_map["block2_conv2"],
        
        "temporal_blocks.1.conv2"                 : base_map["block2_conv2"][4:-4], ## ti
        
        #"BatchNorm4"            : base_map["block3_conv1"],
        #"Relu2"                 : base_map["block3_conv1"],
        
        "temporal_blocks.2.conv1"                 : base_map["block3_conv2"],
        #"BatchNorm5"            : base_map["block3_conv2"],
        
        "temporal_blocks.2.conv2"                 : base_map["block_output"],
        #"BatchNorm6"            : base_map['block_output'],
        #"Relu3"                 : base_map['block_output'],
    }
    
relevant_indices_per_node = create_node_index_map(relevant_indices_per_layer)
print_relevant_indices_map(relevant_indices_per_node)

# save the relevant indices per node
with open("relevant_indices_per_model_layer.json", "w") as f:
    json.dump(relevant_indices_per_node, f, indent=2)