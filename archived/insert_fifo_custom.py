from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from finn.custom_op.fpgadataflow.streamingfifo import StreamingFIFO
from onnx import helper

import uuid

def gen_finn_name(base: str = "node") -> str:
    return base + "_" + str(uuid.uuid4())[:8]


def insert_fifo_before_node(model: ModelWrapper, target_node_name: str, fifo_depth: int = 2) -> ModelWrapper:
    target_node = model.get_node_from_name(target_node_name)
    input_tensor = target_node.input[0]  # only replace the first input (stream input)
    fifo_output_tensor = input_tensor + "_fifo"

    fifo_node = helper.make_node(
        "StreamingFIFO",
        inputs=[input_tensor],
        outputs=[fifo_output_tensor],
        name="FIFO_" + input_tensor,
        domain="finn.custom_op.fpgadataflow"
    )

    index = next(i for i, n in enumerate(model.graph.node) if n.name == target_node_name)
    print(f"{target_node.name} inputs BEFORE FIFO: {target_node.input}")

    model.graph.node.insert(index, fifo_node)

    # Only update input[0], which is the streamed data input
    target_node.input[0] = fifo_output_tensor

    # Copy metadata
    model.set_tensor_shape(fifo_output_tensor, model.get_tensor_shape(input_tensor))
    model.set_tensor_datatype(fifo_output_tensor, model.get_tensor_datatype(input_tensor))
    model.set_tensor_layout(fifo_output_tensor, model.get_tensor_layout(input_tensor))

    fifo_inst = StreamingFIFO(fifo_node)
    fifo_inst.set_nodeattr("depth", fifo_depth)
    fifo_inst.set_nodeattr("dataType", model.get_tensor_datatype(input_tensor).name)

    
    
    return model


def insert_fifos_between_all_fpgadataflow_nodes(model, fifo_depth=2):
    new_nodes = []

    for node in model.graph.node:
        if node.domain != "finn.custom_op.fpgadataflow":
            continue

        for output_name in node.output:
            consumers = model.find_consumers(output_name)

            for consumer_node in consumers:
                if consumer_node.domain != "finn.custom_op.fpgadataflow":
                    continue

                # Only insert if the output connects to input[0] (stream input)
                if consumer_node.input[0] != output_name:
                    print(f"⚠️ Skipping FIFO: {output_name} is not connected to {consumer_node.name}'s stream input (input[0])")
                    continue

                fifo_output_name = output_name + "_fifo"
                fifo_node_name = output_name + "_FIFO"

                # Patch input[0] only
                print(f"Inserting FIFO between {node.name} -> {consumer_node.name}")
                print(f"{consumer_node.name} inputs BEFORE: {consumer_node.input}")
                consumer_node.input[0] = fifo_output_name
                print(f"{consumer_node.name} inputs AFTER: {consumer_node.input}")

                # Create FIFO node
                fifo_node = helper.make_node(
                    "StreamingFIFO",
                    inputs=[output_name],
                    outputs=[fifo_output_name],
                    name=fifo_node_name,
                    domain="finn.custom_op.fpgadataflow",
                )

                # Copy shape and datatype
                model.set_tensor_shape(fifo_output_name, model.get_tensor_shape(output_name))
                model.set_tensor_datatype(fifo_output_name, model.get_tensor_datatype(output_name))
                model.set_tensor_layout(fifo_output_name, model.get_tensor_layout(output_name))

                # Set FIFO attributes
                fifo_inst = StreamingFIFO(fifo_node)
                fifo_inst.set_nodeattr("depth", fifo_depth)
                fifo_inst.set_nodeattr("folded_shape", model.get_tensor_shape(output_name))
                fifo_inst.set_nodeattr("dataType", str(model.get_tensor_datatype(output_name)))

                new_nodes.append(fifo_node)

    model.graph.node.extend(new_nodes)
    return model


from qonnx.custom_op.registry import getCustomOp
from qonnx.core.modelwrapper import ModelWrapper
from onnx import helper


def insert_fifos_from_nonfpgadataflow_to_fpgadataflow_safe(model: ModelWrapper, fifo_depth=2):
    graph_changed = False
    insertions = []

    for node in model.graph.node:
        if node.domain == "finn.custom_op.fpgadataflow":
            continue

        for output_tensor in node.output:
            consumers = model.find_consumers(output_tensor)
            for consumer in consumers:
                if consumer.domain == "finn.custom_op.fpgadataflow":
                    if output_tensor.endswith("_FIFO"):
                        continue
                    insertions.append((node, output_tensor, consumer))

    for node, output_tensor, consumer in insertions:
        fifo_output = output_tensor + "_FIFO"
        fifo_node = helper.make_node(
            "StreamingFIFO",
            [output_tensor],
            [fifo_output],
            name=gen_finn_name("FIFO"),
            domain="qonnx.custom_op.general",  # force partitioning
        )
        model.graph.node.insert(list(model.graph.node).index(consumer), fifo_node)

        fifo_inst = StreamingFIFO(fifo_node)
        fifo_inst.set_nodeattr("depth", fifo_depth)

        model.set_tensor_shape(fifo_output, model.get_tensor_shape(output_tensor))
        model.set_tensor_datatype(fifo_output, model.get_tensor_datatype(output_tensor))
        model.set_tensor_layout(fifo_output, model.get_tensor_layout(output_tensor))

        for idx, inp in enumerate(consumer.input):
            if inp == output_tensor:
                consumer.input[idx] = fifo_output

        print(f"Inserted FIFO between {node.name} -> {consumer.name}")
        graph_changed = True

    return model

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from finn.custom_op.fpgadataflow.streamingfifo import StreamingFIFO
from onnx import helper
import uuid

def gen_finn_name(base: str = "node") -> str:
    return base + "_" + str(uuid.uuid4())[:8]

def insert_fifo_between(model: ModelWrapper, producer, consumer, output_tensor, fifo_depth=2):
    fifo_output = output_tensor + "_FIFO"
    fifo_node = helper.make_node(
        "StreamingFIFO",
        [output_tensor],
        [fifo_output],
        name=gen_finn_name("FIFO"),
        domain="finn.custom_op.fpgadataflow",
    )

    fifo_inst = StreamingFIFO(fifo_node)
    fifo_inst.set_nodeattr("depth", fifo_depth)
    fifo_inst.set_nodeattr("folded_shape", model.get_tensor_shape(output_tensor))
    fifo_inst.set_nodeattr("dataType", str(model.get_tensor_datatype(output_tensor)))
    fifo_inst.set_nodeattr("backend", "fpgadataflow")  # ✅ Required

    model.set_tensor_shape(fifo_output, model.get_tensor_shape(output_tensor))
    model.set_tensor_datatype(fifo_output, model.get_tensor_datatype(output_tensor))
    model.set_tensor_layout(fifo_output, model.get_tensor_layout(output_tensor))

    # Insert FIFO immediately after producer
    node_list = list(model.graph.node)
    insert_index = node_list.index(producer) + 1
    model.graph.node.insert(insert_index, fifo_node)

    # Redirect consumer input to FIFO output
    for idx, inp in enumerate(consumer.input):
        if inp == output_tensor:
            consumer.input[idx] = fifo_output

    print(f"✅ Inserted FIFO between {producer.name} -> {consumer.name}")
    return model

def insert_fifos_for_partition_boundary(model: ModelWrapper, fifo_depth=2):
    """
    Automatically insert FIFOs for all transitions from non-fpgadataflow nodes to fpgadataflow nodes.
    """
    insertions = []

    for node in model.graph.node:
        if node.domain == "finn.custom_op.fpgadataflow":
            continue

        for output_tensor in node.output:
            consumers = model.find_consumers(output_tensor)
            for consumer in consumers:
                if consumer.domain == "finn.custom_op.fpgadataflow":
                    if output_tensor.endswith("_FIFO"):
                        continue
                    insertions.append((node, consumer, output_tensor))

    for producer, consumer, output_tensor in insertions:
        model = insert_fifo_between(model, producer, consumer, output_tensor, fifo_depth)

    return model


from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

def insert_manual_tlast_before_node(model: ModelWrapper, target_node_name: str):
    # Get the node and its first input
    target_node = model.get_node_from_name(target_node_name)
    tlast_input = target_node.input[0]
    tlast_output = tlast_input + "_tlast"

    # Create the TLastMarker node
    tlast_node = helper.make_node(
        "TLastMarker",
        [tlast_input],
        [tlast_output],
        name=gen_finn_name("TLastMarker"),
        domain="finn.custom_op.fpgadataflow"  # ✅ FIXED
    )

    # Register the op by importing and wrapping it
    tlast_inst = TLastMarker(tlast_node)
    tlast_inst.set_nodeattr("NumIters", 1)  # ← Adjust this if needed

    # Insert the node before the target node
    insert_idx = model.graph.node.index(target_node)
    model.graph.node.insert(insert_idx, tlast_node)

    # Set shape/type/layout of the new tensor
    model.set_tensor_shape(tlast_output, model.get_tensor_shape(tlast_input))
    model.set_tensor_datatype(tlast_output, model.get_tensor_datatype(tlast_input))
    model.set_tensor_layout(tlast_output, model.get_tensor_layout(tlast_input))

    # Update target node's input
    target_node.input[0] = tlast_output

    print(f"✅ Inserted TLastMarker before {target_node_name}")
    return model
