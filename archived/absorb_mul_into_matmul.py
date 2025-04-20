from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
import numpy as np

class AbsorbMulIntoMatMul(Transformation):
    """
    Look for patterns of the form:
      MatMul -> Mul(scalar) -> <consumer or graph output>
    and fold the Mul's scalar factor into the MatMul's weight, removing the Mul node.
    """

    def _remove_initializer_and_input(self, model, init_name):
        """Manually remove initializer 'init_name' from model.graph.initializer
        and from model.graph.input if present."""
        # remove from graph.initializer
        for i, init in enumerate(model.graph.initializer):
            if init.name == init_name:
                del model.graph.initializer[i]
                break
        # also remove from graph.input if present
        for i, inp in enumerate(model.graph.input):
            if inp.name == init_name:
                del model.graph.input[i]
                break

    def apply(self, model: ModelWrapper):
        graph_modified = False

        node_list = list(model.graph.node)  # copy so we can modify safely
        for node in node_list:
            if node.op_type != "Mul":
                continue

            mul_node = node
            mul_in0 = mul_node.input[0]
            mul_in1 = mul_node.input[1]

            # find the node that feeds Mul input[0] and Mul input[1]
            producers_0 = model.find_producer(mul_in0)
            producers_1 = model.find_producer(mul_in1)

            # We want exactly one input to be from a MatMul node, the other to be a scalar initializer.
            matmul_node = None
            matmul_input_id = None
            scalar_input_id = None

            # check input0
            if producers_0 is not None and producers_0.op_type == "MatMul":
                matmul_node = producers_0
                matmul_input_id = mul_in0
                scalar_input_id = mul_in1
            # else check input1
            elif producers_1 is not None and producers_1.op_type == "MatMul":
                matmul_node = producers_1
                matmul_input_id = mul_in1
                scalar_input_id = mul_in0
            else:
                # no MatMul -> skip
                continue

            # check if scalar_input_id is truly a scalar initializer
            scalar_init = model.get_initializer(scalar_input_id)
            if scalar_init is None:
                # not an initializer => skip
                continue
            if scalar_init.size != 1:
                # not a scalar => skip
                continue

            scale_factor = float(scalar_init.flatten()[0])

            # find MatMul weight
            matmul_in0 = matmul_node.input[0]
            matmul_in1 = matmul_node.input[1]
            matmul_weight_id = None
            matmul_data_id = None

            if model.get_initializer(matmul_in0) is not None and model.get_initializer(matmul_in1) is None:
                matmul_weight_id = matmul_in0
                matmul_data_id = matmul_in1
            elif model.get_initializer(matmul_in1) is not None and model.get_initializer(matmul_in0) is None:
                matmul_weight_id = matmul_in1
                matmul_data_id = matmul_in0
            else:
                # not recognized => skip
                continue

            W = model.get_initializer(matmul_weight_id)
            if W is None:
                continue

            # fold scale into MatMul weight
            W_new = W * scale_factor
            model.set_initializer(matmul_weight_id, W_new)

            # rewire the Mul output to come directly from MatMul
            mul_out_id = mul_node.output[0]
            consumers = model.find_consumers(mul_out_id)

            if consumers is not None and len(consumers) > 0:
                # If there are consumers, point them to matmul_node's output
                for cn in consumers:
                    for i, inp in enumerate(cn.input):
                        if inp == mul_out_id:
                            cn.input[i] = matmul_node.output[0]
            else:
                # no consumers => the Mul is likely driving the final graph output
                # rename the MatMul output to the Mul's output name so the graph output remains correct
                matmul_node.output[0] = mul_out_id

            # remove the Mul node
            if mul_node in model.graph.node:
                model.graph.node.remove(mul_node)

            # remove the scalar initializer
            self._remove_initializer_and_input(model, scalar_input_id)

            graph_modified = True

        return (model, graph_modified)
