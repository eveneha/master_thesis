import os 
import pandas as pd 

file_path = "/home/eveneiha/MASTER/Data/mitDB/"

aami_classes = {
    'N': ['N', '·', 'L', 'R', 'e', 'j'],
    'S': ['A', 'a', 'J', 'S'],
    'V': ['V', 'E'],
    'F': ['F'],
    'Q': ['/', 'f', 'Q', '[', ']', '!', 'x', '|'],
    '-': ['-']
}

fp_it = next(os.walk(file_path))[2]
fp_it.sort()

for file in fp_it:
    df = pd.read_csv(f"{file_path}{file}")
    # Flatten the aami_classes dictionary into a lookup dictionary
    lookup_dict = {val: key for key, values in aami_classes.items() for val in values}

    # Map the Type column using the lookup dictionary and handle missing values
    df['Type'] = df['Type'].map(lookup_dict).fillna('Q')    
    df.to_csv(f"/home/eveneiha/MASTER/Data/mitDB_AMII{file}", index=False)
    print(f"File: {file} converted")

