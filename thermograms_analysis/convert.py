import numpy as np
from tqdm import tqdm
import os
from utils import readPTWHeader, getPTWFrames


ptws = os.listdir('thermograms_analysis/ptw/')

for ptw in tqdm(ptws, total=len(ptws)):

    header = readPTWHeader(f"thermograms_analysis/ptw/{ptw}")

    frames = getPTWFrames(header, range(1, header.h_lastframe + 1))[0]
    np.save(f"thermograms_analysis/data/thermogram_{ptw.replace('.ptw', '')}.npy", frames.astype(np.int16))