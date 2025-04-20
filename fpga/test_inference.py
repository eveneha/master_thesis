import numpy as np
import time
from driver import FINNExampleOverlay
from qonnx.core.datatype import DataType
# === Load test data ===
inputs = np.load("input.npy").astype(np.int8)      # shape: (N, 1000, 1, 1)
labels = np.load("labels.npy").astype(np.int32)    # shape: (N,)
print("✅ Test data loaded.")
print("Input shape:", inputs.shape)
print("Labels shape:", labels.shape)

# === Define platform and shapes ===
platform = "zynq-iodma"
io_shape_dict = {
    "idt": [DataType["INT8"]],
    "odt": [DataType["INT18"]],
    "ishape_normal": [(1, 1000, 1, 1)],
    "oshape_normal": [(1, 1, 1, 5)],
    "ishape_folded": [(1, 1000, 1, 1, 1)],
    "oshape_folded": [(1, 1, 1, 5, 1)],
    "ishape_packed": [(1, 1000, 1, 1, 1)],
    "oshape_packed": [(1, 1, 1, 5, 3)],
    "input_dma_name": ['idma0'],
    "output_dma_name": ['odma0'],
    "number_of_external_weights": 0,
    "num_inputs": 1,
    "num_outputs": 1,
}

print("Entering FINNEcampleOverlay")

ol = FINNExampleOverlay("resizer.bit", platform=platform, io_shape_dict=io_shape_dict)
print("Available IP blocks:", ol.ip_dict.keys())
for name, ip in ol.ip_dict.items():
    print(f"🔧 IP Block: {name}")
    try:
        print("  CTRL Reg [0x00]:", hex(ip.read(0x00)))
    except Exception as e:
        print("  ⚠️ Failed to read CTRL register:", e)


print("IDMA handles:", ol.idma)
print("ODMA handles:", ol.odma)


ol.download()
print("✅ Overlay loaded.")

# === Initialize DMA handles ===
print("Init DMA handles...")
idma = ol.idma0
odma = ol.odma0


# === Run inference ===
print("Running inference...")
correct = 0
total = inputs.shape[0]
latencies = []
print(inputs[1].shape)
for i in range(total):
    x = inputs[i]  # already shape (1, 1000, 1, 1)
    x = x.reshape(1, 1000, 1, 1)  # add batch dim → (1, 1000, 1, 1)
    label = labels[i]

    start = time.time()

    output = ol.execute(x)

    end = time.time()

    latency_ms = (end - start) * 1000
    latencies.append(latency_ms)

    pred = np.argmax(output)
    if pred == label:
        correct += 1

    if i % 200 == 0:
        print(f"[{i}/{total}] Label: {label}, Prediction: {pred}, Latency: {latency_ms:.2f} ms")


# === Report accuracy and timing ===
acc = correct / total
avg_latency = np.mean(latencies)
print(f"\n🎯 Accuracy: {acc * 100:.2f}% ({correct}/{total})")
print(f"⏱️  Average inference time: {avg_latency:.2f} ms per sample")
