import os
import pandas as pd

def find_label_at_sample(data_dir, target_index=12861):
    """
    Search all CSV files in the specified directory for a label
    corresponding to a given sample index.
    
    Args:
        data_dir (str): Path to directory containing CSV files
        target_index (int): Sample index to search for (default: 12861)
    """
    found = False
    for fname in os.listdir(data_dir):
        if fname.endswith('.csv'):
            fpath = os.path.join(data_dir, fname)
            try:
                df = pd.read_csv(fpath)
                if 'Sample' in df.columns and 'Type' in df.columns:
                    hits = df[df['Sample'] == target_index]
                    if not hits.empty:
                        print(f"✅ Found in: {fname}")
                        print(hits[['Sample', 'Type']])
                        found = True
            except Exception as e:
                print(f"⚠️ Failed to read {fname}: {e}")
    
    if not found:
        print(f"❌ No label found at index {target_index} in any file.")

# Example usage:
find_label_at_sample("/home/eveneiha/finn/workspace/ml/data/MITDB/merged_output/")
