import numpy as np
from tqdm import tqdm
import glob
from utils import readPTWHeader, getPTWFrames


ptws = glob.glob('*.ptw')

for ptw in tqdm(ptws, total=len(ptws)):

    header = readPTWHeader(ptw)

    frames = getPTWFrames(header, range(1, header.h_lastframe + 1))[0]
    np.save(f"data/thermogram_{ptw.replace('.ptw', '')}.npy", frames.astype(np.int16))