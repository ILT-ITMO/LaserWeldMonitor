from modules import process_thermogram
from tqdm import tqdm
import os
import json

W_SIZE = 40

paths = os.listdir('thermograms_analysis/data/')
counts = {}

# for path in tqdm(paths):
#    counts[path] = process_thermogram(f"data/{path}", W_SIZE)

#process_thermogram('thermograms_analysis/data/thermogram_7.npy', 10)



# with open(f"metrics_{W_SIZE}_lstm.json", 'w') as f:
#     json.dump(counts, f)

print(counts)