import onnx
from onnx import helper

CENTER_INDEX = 84

def load_model(path):
    return onnx.load(path)

def get_tensor_shape(model, tensor_name):
    for vi in model.graph.value_info:
        if vi.name == tensor_name:
            return [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
    for vi in model.graph.output:
        if vi.name == tensor_name:
            return [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
    return None

def get_attr(node, name, default):
    for attr in node.attribute:
        if attr.name == name:
            if attr.ints:
                return list(attr.ints)[0]
            elif attr.i:
                return attr.i
    return default

def trace_back_dependencies(model, output_tensor, target_index):
    keep_nodes = set()
    visited = set()
    worklist = [(output_tensor, target_index)]

    while worklist:
        tensor_name, idx = worklist.pop()
        if (tensor_name, idx) in visited:
            continue
        visited.add((tensor_name, idx))

        for node in model.graph.node:
            if tensor_name in node.output:
                keep_nodes.add(node.name)
                op_type = node.op_type
                print(f"📌 Tracing node {node.name} ({op_type}) for index {idx}")

                if op_type in ["Conv", "QuantConv"]:
                    in_tensor = node.input[0]
                    input_shape = get_tensor_shape(model, in_tensor)
                    if input_shape is None:
                        continue
                    H = input_shape[2]

                    kernel_size = get_attr(node, "kernel_shape", 9)
                    dilation = get_attr(node, "dilations", 1)
                    stride = get_attr(node, "strides", 1)

                    eff_k = (kernel_size - 1) * dilation + 1
                    start = idx * stride
                    input_indices = [start + i * dilation for i in range(kernel_size)]
                    input_indices = [i for i in input_indices if 0 <= i < H]

                    for iidx in input_indices:
                        worklist.append((in_tensor, iidx))

                elif op_type in ["BatchNormalization", "Relu", "Quant"]:
                    in_tensor = node.input[0]
                    worklist.append((in_tensor, idx))

                else:
                    print(f"⚠️ Unhandled op type: {op_type} — skipping.")

    return keep_nodes

def main():
    model_path = "/home/eveneiha/finn/workspace/finn/onnx/tcn_v41.onnx"
    model = load_model(model_path)

    # Find the input tensor to Slice_0
    slice_node = next((n for n in model.graph.node if n.op_type == "Slice"), None)
    if slice_node is None:
        print("❌ No Slice node found!")
        return

    slice_input_tensor = slice_node.input[0]

    # Trace back from slice input tensor at CENTER_INDEX
    keep = trace_back_dependencies(model, slice_input_tensor, CENTER_INDEX)

    all_nodes = set(n.name for n in model.graph.node)
    prune = sorted(all_nodes - keep)
    keep = sorted(keep)

    print("\n✅ Nodes to KEEP:")
    for n in keep:
        print(f"  🔹 {n}")

    print("\n🗑️ Nodes to PRUNE:")
    for n in prune:
        print(f"  ❌ {n}")

if __name__ == "__main__":
    main()
