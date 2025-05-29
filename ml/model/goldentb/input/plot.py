import matplotlib.pyplot as plt

def plot_ecg_from_txt(filepath, title="ECG Signal", figsize=(12, 4)):
    """
    Load ECG values from a .txt file and plot them.

    The file should contain one numeric value per line.

    Parameters:
    - filepath: Path to the .txt file.
    - title: Plot title.
    - figsize: Figure size.
    """
    try:
        with open(filepath, 'r') as f:
            values = [int(line.strip()) for line in f if line.strip()]
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    if not values:
        print("⚠️ File is empty or invalid format.")
        return

    plt.figure(figsize=figsize)
    plt.plot(values, linewidth=1)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
plot_ecg_from_txt("/home/eveneiha/finn/workspace/ml/model/goldentb/input/tb_input_data.txt")
