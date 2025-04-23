import json

from qonnx.core.modelwrapper import ModelWrapper
# Not using PruneChannels from qonnx.transformation.pruning anymore in the main prune step

from prune_time_indices import PruneSamples, InsertConvGatherNodes # Import your custom transformations

# from qonnx.transformation import ConvertQONNXtoFINN # Not used here
# from qonnx.core.modelwrapper import ModelWrapper # Already imported

from finn.transformation.qonnx.fold_quant_weights import FoldQuantWeights
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames, RemoveUnusedTensors
from qonnx.transformation.infer_shapes import InferShapes # Import InferShapes for robustness

from qonnx.transformation.general import RemoveUnusedTensors 

from onnx import helper, AttributeProto

def force_dilation_to_1(model):
    for node in model.graph.node:
        if node.op_type == "Conv":
            for attr in node.attribute:
                if attr.name == "dilations":
                    attr.ints[:] = [1, 1]
                    print(f"🔁 Set dilation=1 for Conv node: {node.name}")
    return model




# Load model
# Use a cleaner path variable
onnx_model_path = "/home/eveneiha/finn/workspace/finn/onnx/tcn_v41.onnx"
model = ModelWrapper(onnx_model_path)
print(f"Loaded model from: {onnx_model_path}")



# Apply cleanup transformations
print("Applying cleanup transformations...")
# FoldQuantWeights might be important for FINN compatibility if weights are quantized
model = model.transform(FoldQuantWeights())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
print("Cleanup transformations applied.")


#new_shape = [1, 1, 665, 1]
#model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 665


# === REMOVE Slice node ===
print("Checking for and removing Slice nodes...")
# Create a copy of the node list to iterate over while modifying
slice_nodes = list(model.graph.node) # Get current nodes
slice_nodes_to_remove = [n for n in slice_nodes if n.op_type == "Slice"]

if not slice_nodes_to_remove:
    print("✅ No Slice node found, skipping removal.")
else:
    for slice_node in slice_nodes_to_remove:
        slice_input = slice_node.input[0]
        slice_output = slice_node.output[0]

        # Find all nodes that consume the output of the slice node
        nodes_to_rewire = []
        # Iterate through the *current* nodes in the graph to find consumers
        current_nodes_list = list(model.graph.node)
        for node in current_nodes_list:
            if slice_output in node.input:
                 nodes_to_rewire.append(node)

        if nodes_to_rewire:
            for node in nodes_to_rewire:
                 for i, inp in enumerate(node.input):
                    if inp == slice_output:
                        node.input[i] = slice_input
                        print(f"🔁 Rewired {node.name} input {i} to bypass {slice_node.name}")

        # Remove the slice node itself
        model.graph.node.remove(slice_node)
        print(f"❌ Removed Slice node: {slice_node.name}")

# Save the cleaned up model before pruning
cleaned_model_path = "model_cleaned_no_slice.onnx"
model.save(cleaned_model_path)
print(f"Cleaned model saved to: {cleaned_model_path}")


# Load your prune_spec
prune_spec_path = "prune_spec.json"
try:
    with open(prune_spec_path, "r") as f:
        prune_spec = json.load(f)
    print(f"Loaded prune_spec from: {prune_spec_path}")
except FileNotFoundError:
    print(f"Error: {prune_spec_path} not found. Please generate it first.")
    exit() # Exit if prune spec is missing
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {prune_spec_path}. Check file format.")
    exit() # Exit if JSON is invalid
    
    
# Load your prune_spec
prune_spec_path = "gather_map.json"
try:
    with open(prune_spec_path, "r") as f:
        gather_map = json.load(f)
    print(f"Loaded prune_spec from: {prune_spec_path}")
except FileNotFoundError:
    print(f"Error: {prune_spec_path} not found. Please generate it first.")
    exit() # Exit if prune spec is missing
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {prune_spec_path}. Check file format.")
    exit() # Exit if JSON is invalid


print("Running temporal pruning...")
# Now prune, set lossy=True because lossless may break (as warned)
# *** WORKAROUND FOR TypeError: cannot unpack non-iterable ModelWrapper object ***
# Instead of: pruned_model, _ = model.transform(PruneSamples(...))
# Just assign the result directly. This suggests model.transform might return
# just the model object in some scenarios, contrary to the expected tuple.
# The PruneSamples transformation itself should handle its internal rerunning logic.
pruned_model = model.transform(PruneSamples(prune_spec=prune_spec, lossy=True, stop_node_name="Quant_10"))
print("Temporal pruning done.")


# --- Reduce model input tensor shape to 665 ---
print("Reducing model input shape to 883 samples...")

input_tensor = pruned_model.graph.input[0]
# Set temporal length (dim[2]) to 665
input_tensor.type.tensor_type.shape.dim[2].dim_value = 833


# --- Apply Step 2: Insert Gather after specific Conv outputs ---






# print("Running Step 2: Inserting Gather nodes after Conv outputs...")
# # This uses gather_map (relevant indices for specific conv outputs)
# model = model.transform(InsertConvGatherNodes(relevant_indices_map=gather_map)) # Stop node might not be needed here if map is correct
# print("Step 2 finished.")







# # Find the Conv_1 node
# conv1_node = next(n for n in pruned_model.graph.node if n.name == "Conv_1")

# # # Get the input tensor name
# conv1_input_tensor_name = conv1_node.input[0]

# # # Get and print the shape
# # conv1_input_shape = pruned_model.get_tensor_shape(conv1_input_tensor_name)
# # print(f"🔍 Conv_1 expects input tensor '{conv1_input_tensor_name}' with shape: {conv1_input_shape}")

# conv1_output_tensor_name = conv1_node.output[0]
# # conv1_output_shape = model.get_tensor_shape(conv1_output_tensor_name)
# # print(f"📤 Conv_1 produces output tensor '{conv1_output_tensor_name}' with shape: {conv1_output_shape}")

# print("Conv_1 input shape:", pruned_model.get_tensor_shape(conv1_input_tensor_name))
# print("Conv_1 output shape:", pruned_model.get_tensor_shape(conv1_output_tensor_name))


for inp in model.model.graph.input:
    inp.type.tensor_type.shape.ClearField("dim")

# After pruning and shape updates, run InferShapes as a final step for consistency
print("Running final shape inference...")
try:
    pruned_model = pruned_model.transform(InferShapes())
    print("Shape inference completed.")
except Exception as e:
    print(f"Warning: Shape inference failed after pruning: {e}")
    print("The pruned model might have inconsistent shapes.")

print("Forcing dilation to 1...")
pruned_model = force_dilation_to_1(pruned_model)

# Save result
pruned_model_path = "pruned_model.onnx"
pruned_model.save(pruned_model_path)
print(f"Pruned model saved to: {pruned_model_path}")

