import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from init_model import ol  # assumes your overlay is initialized there

# ——— load your data once ———
inputs = np.load("input.npy").astype(np.float32)   # (N,1000,1,1)
labels = np.load("labels.npy").astype(np.int32) # (N,)

inputs1 = np.load("pynq_input1000.npy")
inputs1 = np.array(inputs1)



##--- Quantize Inputs ---##
# 0.007948528975248337
# scale=0.008124164596665651
# 0.007684304844588041
#0.007885444909334183
#scale=0.007684304844588041

# 0.0405256450176239
def quantize_input(x_float, scale=0.007885444909334183, zero_point=0):
    x_int = np.clip(np.round(x_float / scale) + zero_point, -128, 127)
    return x_int.astype(np.int8)

inputs = [quantize_input(x) for x in inputs]
inputs = np.array(inputs)

# ——— inference loop ———
correct = 0
total   = inputs.shape[0]
latencies = []
all_preds  = []
all_targets= []
misclassified = defaultdict(lambda: defaultdict(list))



sample_input_for_fpga = inputs[0].reshape(1, 1000, 1, 1)


def int8_to_bytes(arr: np.ndarray) -> bytes:
  
    if arr.dtype != np.int8:
        raise ValueError("Expected dtype=int8")
    return arr.astype(np.uint8).tobytes()

print("▶️  Starting inference loop…")
#print(inputs2.shape)
for i in range(total): #total
    x = inputs[i].reshape(1,1000,1,1)
    x = x[:,168:833,:,:]
    y = int(labels[i])
    t0 = time.time()
    out = ol.execute(x)
    t1 = time.time()


    if i % 7500 == 0:
        print(f"\n--- Batch {i} ---")
        print(f"Rawtput array (flattened): {out}")
        print(f"Raw output min: {np.min(out)}, max: {np.max(out)}")


    lat_ms = (t1 - t0) * 1000
    latencies.append(lat_ms)

    pred = int(np.argmax(out))
    all_preds.append(pred)
    all_targets.append(y)
    if pred == y:
        correct += 1
    else:
        misclassified[pred][y].append(i)

    if i %  7500 == 0:
        print(f"  [{i}/{total}] lab={y}  pred={pred}  lat={lat_ms:.2f} ms")

# ——— summary ———
acc = correct / total
lat = np.array(latencies)
print(f"\n🎯 Accuracy: {acc*100:.2f}%")
print(f"⏱  Latency (ms): min {lat.min():.2f}, med {np.median(lat):.2f}, 90% {np.percentile(lat,90):.2f}, max {lat.max():.2f}")


# ——— confusion matrix ———
num_classes = max(max(all_targets), max(all_preds)) + 1
cm = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(all_targets, all_preds):
    cm[t, p] += 1

plt.figure(figsize=(6,6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, cm[i,j],
                 ha="center", va="center",
                 color="white" if cm[i,j] > cm.max()/2 else "black")
plt.xticks(range(num_classes))
plt.yticks(range(num_classes))
plt.tight_layout()
plt.savefig("confusion_matrix.png")

# ——— per‑class precision & recall ———
print("\n📊 Per‑class precision & recall:")
for i in range(num_classes):
    tp = cm[i,i]
    fp = cm[:,i].sum() - tp
    fn = cm[i,:].sum() - tp
    prec = tp / (tp + fp) if (tp + fp)>0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn)>0 else 0.0
    print(f"  Class {i:2d}: precision={prec:.3f}, recall={rec:.3f}")

# ——— F1 scores ———
f1_scores = []
supports  = cm.sum(axis=1)  # number of true samples per class
print("\n🎯 Per‑class F1 scores:")
for i in range(num_classes):
    tp = cm[i,i]
    fp = cm[:,i].sum() - tp
    fn = cm[i,:].sum() - tp
    prec = tp / (tp + fp) if (tp + fp)>0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn)>0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec)>0 else 0.0
    f1_scores.append(f1)
    print(f"  Class {i:2d}: F1 = {f1:.3f}  (support={supports[i]})")

# Macro‑F1 (unweighted mean)
macro_f1 = np.mean(f1_scores)

# Weighted‑F1 (support‑weighted mean)
total_support = supports.sum()
weighted_f1 = np.sum(f1_scores * (supports / total_support))

print(f"\n📊 Avg. Macro‑F1:    {macro_f1:.3f}")
print(f"📊 Weighted‑F1: {weighted_f1:.3f}")


# ——— misclassification breakdown ———
print("\n🚩 Misclassifications:")
for p, true_map in misclassified.items():
    for t, idxs in true_map.items():
        print(f"  Pred {p} → True {t}: {len(idxs)} samples")
