from qonnx.core.modelwrapper import ModelWrapper
import numpy as np
from onnx import numpy_helper

def print_weight_stats(model_path):
    model = ModelWrapper(model_path)
    for init in model.model.graph.initializer:
        if "param0" in init.name:
            weights = numpy_helper.to_array(init)
            print(f"{init.name}: mean={np.mean(weights):.4f}, std={np.std(weights):.4f}")

print("📦 v33 (possibly untrained):")
print_weight_stats("./onnx/tcn_v33.onnx")

print("\n📦 v34 (supposedly trained):")
print_weight_stats("./onnx/tcn_v34.onnx")
