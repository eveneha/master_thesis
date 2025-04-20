import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import torch
import os 
from torch.utils.data import Dataset, DataLoader, Subset
from qonnx.util.cleanup import cleanup_model
from qonnx.core.modelwrapper import ModelWrapper



save_dir = "/home/eveneiha/finn/workspace/ml/data/"

# Load test dataset 
test_data = torch.load(os.path.join(save_dir, "test.pt"))
test_inputs = test_data["inputs"]
test_labels = test_data["labels"]
test_ids = test_data["window_ids"]

class PreprocessedECGDataset(Dataset):
    def __init__(self, inputs, labels, win_ids):
        self.inputs = inputs
        self.labels = labels
        self.win_ids = win_ids

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.win_ids[idx]


test_dataset  = PreprocessedECGDataset(test_inputs, test_labels, test_ids)
batch_size = 16
test_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = ModelWrapper("pruned_model.onnx")
model = cleanup_model(model, preserve_qnt_ops=False)
model.save("pruned_model_cleaned.onnx")


# Load the ONNX pruned model
session = ort.InferenceSession("pruned_model_cleaned.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Metrics
misclassified_map = defaultdict(lambda: defaultdict(list))
all_preds = []
all_targets = []
correct = 0
total = 0

misclassified_inputs = []
misclassified_preds = []
misclassified_targets = []

# No test loss from ONNX model unless you're simulating it
print("🚀 Running ONNX evaluation...")
for inputs_np, targets_np, win_ids in test_dataloader:
    outputs = session.run(None, {input_name: inputs_np})[0]
    preds = np.argmax(outputs, axis=1)

    all_preds.extend(preds)
    all_targets.extend(targets_np)

    correct += np.sum(preds == targets_np)
    total += len(targets_np)

    for i in range(len(targets_np)):
        if preds[i] != targets_np[i]:
            misclassified_inputs.append(inputs_np[i])
            misclassified_preds.append(preds[i])
            misclassified_targets.append(targets_np[i])

            pred_label = AAMI_CLASS_NAMES[preds[i]]
            true_label = AAMI_CLASS_NAMES[targets_np[i]]
            win_id = win_ids[i]
            misclassified_map[pred_label][true_label].append(win_id)

test_accuracy = correct / total
print(f"\n📈 ONNX Test Accuracy: {test_accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=AAMI_CLASS_NAMES,
            yticklabels=AAMI_CLASS_NAMES)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (ONNX Model)")
plt.show()

# Classification Report
print("📋 Classification Report:")
print(classification_report(all_targets, all_preds, target_names=AAMI_CLASS_NAMES, zero_division=0))

# F1 Scores
macro_f1 = f1_score(all_targets, all_preds, average="macro")
weighted_f1 = f1_score(all_targets, all_preds, average="weighted")
print(f"📊 Macro F1-Score: {macro_f1:.4f}")
print(f"📊 Weighted F1-Score: {weighted_f1:.4f}")
