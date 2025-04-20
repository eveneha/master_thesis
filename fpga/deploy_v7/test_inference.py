import numpy as np
import time
from driver import FINNExampleOverlay

# === Load test data ===
inputs = np.load("input.npy").astype(np.int8)      # shape: (N, 1000, 1, 1)
labels = np.load("labels.npy").astype(np.int32)    # shape: (N,)
print("✅ Test data loaded.")
print("Input shape:", inputs.shape)
print("Labels shape:", labels.shape)

# === Define platform and shapes ===
platform = "zynq-iodma"
io_shape_dict = {
    "inputs": {"global_in": (1, 1000, 1, 1)},       # NHWC
    "outputs": {"global_out": (1, 1, 1, 5)},        # NHWC
    "ishape_packed": [(1, 1000, 1, 1)],
    "oshape_packed": [(1, 1, 1, 5)],
    "num_inputs": 1,
    "num_outputs": 1
}

ol = FINNExampleOverlay("resizer.bit", platform=platform, io_shape_dict=io_shape_dict)


ol.download()
print("✅ Overlay loaded.")

# === Initialize DMA handles ===
idma = ol.idma0
odma = ol.odma0

# === Run inference ===
correct = 0
total = inputs.shape[0]
latencies = []

for i in range(total):
    x = inputs[i].reshape(1, 1000, 1, 1)
    label = labels[i]
    out_buffer = np.empty((1, 1, 1, 5), dtype=np.int32)

    start_time = time.time()

    idma.sendchannel.transfer(x)
    odma.recvchannel.transfer(out_buffer)
    idma.sendchannel.wait()
    odma.recvchannel.wait()

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    latencies.append(latency_ms)

    pred = np.argmax(out_buffer)

    if pred == label:
        correct += 1

    if i % 50 == 0:
        print(f"[{i}/{total}] Label: {label}, Prediction: {pred}, Latency: {latency_ms:.2f} ms")

# === Report accuracy and timing ===
acc = correct / total
avg_latency = np.mean(latencies)
print(f"\n🎯 Accuracy: {acc * 100:.2f}% ({correct}/{total})")
print(f"⏱️  Average inference time: {avg_latency:.2f} ms per sample")
