import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file and strip extra quotes from column names
filename = '/home/eveneiha/finn/workspace/ml/data/MITDB/114.csv'
df = pd.read_csv(filename)
df.columns = df.columns.str.strip("'").str.strip('"')

# Show available columns (excluding 'sample #')
columns_to_plot = [col for col in df.columns if col != 'sample #']

# Let the user choose a column
print("Available columns to plot:", columns_to_plot)
selected_column = input(f"Enter the column to plot from {columns_to_plot}: ")

if selected_column not in columns_to_plot:
    print(f"Invalid selection. Please choose from {columns_to_plot}.")
else:
    # Plot the first 1000 samples
    plt.figure(figsize=(12, 6))
    plt.plot(df['sample #'][:1000], df[selected_column][:1000])
    plt.title(f'First 1000 Samples of {selected_column}')
    plt.xlabel('Sample #')
    plt.ylabel(selected_column)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
