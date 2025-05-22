from modules import process_thermogram
from tqdm import tqdm
import os
import json

W_SIZE = 35

#paths = os.listdir('thermograms_analysis/data/')
paths = [f"thermogram_{i}.npy" for i in range(6, 35)]
print(paths)
counts = {}

for path in tqdm(paths):
    if os.path.exists(f"thermograms_analysis/data/{path}"):
        counts[path] = process_thermogram(f"thermograms_analysis/data/{path}", W_SIZE)

#process_thermogram('thermograms_analysis/data/thermogram_7.npy', 10)



with open(f"thermograms_analysis/metrics/metrics_new_{W_SIZE}.json", 'w') as f:
    json.dump(counts, f)

print(counts)