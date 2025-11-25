import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, TensorDataset
from torch.optim import Adam
from torch import nn
import numpy as np
from typing import List, Dict, Tuple, Literal
from math import floor
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_recall_fscore_support
import json
import random
import os
import torch.optim as optim
import cv2

import pandas as pd


QUALITY = {
    "thermogram_1.npy": 
    [[0.00, 0.14, 0.15, 0.00, 0.39, 0.15, 0.00],
    [0.00, 0.16, 0.00, 0.00, 0.27, 0.05, 0.00],
    [0.00, 0.13, 0.00, 0.00, 0.24, 0.02, 0.00],
    [0.00, 0.90, 0.00, 0.00, 0.50, 0.05, 0.00]],
    "thermogram_2.npy": 
    [[0.00, 0.00, 0.04, 0.11, 0.00, 0.21, 0.00],
    [0.00, 0.00, 0.00, 0.13, 0.23, 0.10, 0.00],
    [0.00, 0.00, 0.00, 0.15, 0.13, 0.13, 0.00],
    [0.04, 0.12, 0.03, 0.03, 0.00, 0.19, 0.00]],
    "thermogram_3.npy": 
    [[0.07, 0.10, 0.00, 0.18, 0.00, 0.29, 0.00],
    [0.05, 0.29, 0.02, 0.00, 0.00, 0.23, 0.08],
    [0.00, 0.10, 0.10, 0.00, 0.18, 0.17, 0.00],
    [0.00, 0.17, 0.00, 0.00, 0.46, 0.13, 0.00]],
    "thermogram_4.npy": 
    [[0.12, 0.00, 0.00, 0.14, 0.00, 0.15, 0.00],
    [0.08, 0.09, 0.00, 0.05, 0.00, 0.14, 0.00],
    [0.14, 0.05, 0.05, 0.11, 0.00, 0.15, 0.00],
    [0.11, 0.12, 0.11, 0.06, 0.00, 0.11, 0.00]],
    "thermogram_5.npy": 
    [[0.00, 0.17, 0.00, 0.00, 0.15, 0.10, 0.00],
    [0.10, 0.16, 0.00, 0.07, 0.10, 0.03, 0.00],
    [0.00, 0.25, 0.00, 0.07, 0.23, 0.10, 0.00],
    [0.00, 0.23, 0.00, 0.09, 0.20, 0.09, 0.00]],
    "thermogram_6.npy": 
    [[0.000, 0.180, 0.000, 0.000, 0.174, 0.098, 0.000],
    [0.099, 0.173, 0.097, 0.067, 0.000, 0.025, 0.000],
    [0.000, 0.232, 0.000, 0.081, 0.216, 0.024, 0.000],
    [0.000, 0.085, 0.000, 0.115, 0.134, 0.067, 0.000]],
    "thermogram_7.npy": 
    [[0.091, 0.078, 0.000, 0.132, 0.091, 0.080, 0.000],
    [0.000, 0.132, 0.000, 0.080, 0.134, 0.019, 0.000],
    [0.000, 0.106, 0.000, 0.035, 0.329, 0.037, 0.000],
    [0.000, 0.107, 0.000, 0.032, 0.143, 0.088, 0.000]],
    "thermogram_8.npy":
    [[0.000, 0.126, 0.000, 0.000, 0.147, 0.104, 0.000],
    [0.000, 0.158, 0.000, 0.000, 0.242, 0.027, 0.000],
    [0.000, 0.104, 0.000, 0.000, 0.245, 0.073, 0.000],
    [0.000, 0.104, 0.000, 0.094, 0.239, 0.100, 0.000]],
    "thermogram_9.npy":
    [[0.000, 0.000, 0.000, 0.244, 0.121, 0.046, 0.000],
    [0.000, 0.089, 0.000, 0.000, 0.161, 0.046, 0.000],
    [0.000, 0.049, 0.000, 0.000, 0.129, 0.159, 0.000],
    [0.000, 0.082, 0.000, 0.000, 0.114, 0.221, 0.000]],
    "thermogram_10.npy":
    [[0.000, 0.100, 0.000, 0.000, 0.283, 0.055, 0.000],
    [0.000, 0.189, 0.000, 0.000, 0.264, 0.049, 0.000],
    [0.000, 0.109, 0.000, 0.000, 0.102, 0.174, 0.000],
    [0.000, 0.082, 0.078, 0.078, 0.053, 0.217, 0.000],
    ],
    "thermogram_11.npy":
    [[0.066, 0.067, 0.029, 0.109, 0.000, 0.066, 0.000],
    [0.194, 0.196, 0.000, 0.000, 0.000, 0.040, 0.000],
    [0.000, 0.138, 0.000, 0.136, 0.151, 0.032, 0.000],
    [0.090, 0.075, 0.000, 0.046, 0.000, 0.149, 0.000],
    ],
    "thermogram_12.npy":
    [[0.060, 0.089, 0.000, 0.000, 0.000, 0.015, 0.000],
    [0.088, 0.060, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.127, 0.009, 0.000, 0.073, 0.000, 0.009, 0.000],
    [0.115, 0.083, 0.000, 0.000, 0.000, 0.122, 0.000],
    ],
    "thermogram_13.npy":
    [[0.086, 0.000, 0.000, 0.000, 0.000, 0.187, 0.420],
    [0.103, 0.000, 0.000, 0.000, 0.000, 0.049, 0.000],
    [0.119, 0.087, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.088, 0.063, 0.000, 0.000, 0.000, 0.030, 0.000],
    ],
    "thermogram_14.npy":
    [[0.167, 0.061, 0.000, 0.055, 0.000, 0.000, 0.000],
    [0.166, 0.083, 0.000, 0.076, 0.000, 0.000, 0.000],
    [0.131, 0.077, 0.000, 0.082, 0.000, 0.182, 0.000],
    [0.104, 0.093, 0.092, 0.051, 0.000, 0.360, 0.000],
    ],
    "thermogram_15.npy":
    [[0.128, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.134, 0.151, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.070, 0.153, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.066, 0.256, 0.066, 0.000]],
    "thermogram_16.npy":
    [[0.000, 0.000, 0.000, 0.138, 0.192, 0.050, 0.000],
    [0.000, 0.000, 0.000, 0.069, 0.121, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.181, 0.206, 0.181, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.236, 0.388, 0.000],
    ],
    "thermogram_17.npy":
    [[0.000, 0.000, 0.000, 0.000, 0.100, 0.149, 0.943],
    [0.000, 0.000, 0.000, 0.000, 0.124, 0.146, 0.889],
    [0.000, 0.000, 0.000, 0.000, 0.149, 0.045, 0.210],
    [0.000, 0.000, 0.000, 0.000, 0.140, 0.050, 0.205],
    ],
    "thermogram_18.npy":
    [[0.000, 0.000, 0.077, 0.000, 0.000, 0.060, 0.359],
    [0.000, 0.000, 0.000, 0.000, 0.111, 0.091, 0.401],
    [0.000, 0.110, 0.063, 0.000, 0.063, 0.025, 0.000],
    [0.000, 0.000, 0.032, 0.173, 0.096, 0.073, 0.000],
    ],
    "thermogram_19.npy":
    [[0.000, 0.000, 0.000, 0.000, 0.050, 0.219, 2.101],
    [0.000, 0.000, 0.000, 0.000, 0.078, 0.000, 1.917],
    [0.000, 0.000, 0.000, 0.000, 0.037, 0.000, 1.940],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.196, 1.876],
    ],
    "thermogram_20.npy":
    [[0.000, 0.000, 0.000, 0.000, 0.275, 0.023, 0.494],
    [0.000, 0.000, 0.000, 0.000, 0.317, 0.046, 0.136],
    [0.000, 0.000, 0.000, 0.132, 0.382, 0.026, 0.000],
    [0.000, 0.000, 0.000, 0.086, 0.304, 0.100, 0.000],
    ],
    "thermogram_21.npy":
    [[0.000, 0.000, 0.000, 0.059, 0.122, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.084, 0.131, 0.046, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.224, 0.030, 0.000],
    [0.000, 0.000, 0.000, 0.063, 0.249, 0.000, 0.000]],
    "thermogram_22.npy":
    [[0.000, 0.096, 0.052, 0.000, 0.000, 0.176, 0.000],
    [0.000, 0.050, 0.000, 0.000, 0.127, 0.096, 0.000],
    [0.000, 0.000, 0.000, 0.084, 0.173, 0.000, 0.000],
    [0.000, 0.122, 0.000, 0.000, 0.094, 0.038, 0.000],
    ],
    "thermogram_23.npy":
    [[0.000, 0.176, 0.000, 0.000, 0.176, 0.119, 0.000],
    [0.000, 0.080, 0.000, 0.000, 0.200, 0.044, 0.000],
    [0.000, 0.118, 0.000, 0.000, 0.201, 0.078, 0.000],
    [0.000, 0.325, 0.000, 0.000, 0.259, 0.175, 0.000],
    ],
    "thermogram_24.npy":
    [[0.000, 0.156, 0.000, 0.000, 0.231, 0.140, 0.000],
    [0.000, 0.065, 0.000, 0.000, 0.236, 0.225, 0.000],
    [0.000, 0.072, 0.000, 0.034, 0.323, 0.146, 0.000],
    [0.000, 0.082, 0.000, 0.000, 0.238, 0.183, 0.000],
    ],
    "thermogram_25.npy":
    [[0.000, 0.103, 0.055, 0.000, 0.138, 0.218, 0.000],
    [0.000, 0.137, 0.000, 0.000, 0.119, 0.014, 0.000],
    [0.000, 0.163, 0.000, 0.000, 0.174, 0.000, 0.000],
    [0.000, 0.221, 0.000, 0.000, 0.103, 0.042, 0.000],
    ],
    "thermogram_26.npy":
    [[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.660],
    [0.000, 0.000, 0.000, 0.000, 0.102, 0.000, 1.650],
    [0.000, 0.000, 0.480, 0.000, 0.000, 0.000, 1.926],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.844],
    ],
    # "thermogram_27.npy":
    # [[0.000, 0.000, 0.000, 0.000, 0.088, 0.365, 1.334],
    # [0.000, 0.000, 0.000, 0.000, 0.122, 0.083, 0.582],
    # [0.000, 0.000, 0.000, 0.000, 0.091, 0.097, 0.561],
    # [0.000, 0.000, 0.000, 0.000, 0.072, 0.348, 0.750],
    # ],
    "thermogram_28.npy":
    [[0.000, 0.000, 0.000, 0.000, 0.140, 0.000, 0.350],
    [0.000, 0.000, 0.063, 0.000, 0.000, 0.048, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.070, 0.181, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.112, 0.000, 0.000]],
    "thermogram_29.npy":
    [[0.000, 0.000, 0.000, 0.150, 0.128, 0.270, 0.000],
    [0.000, 0.160, 0.000, 0.118, 0.214, 0.250, 0.000],
    [0.000, 0.177, 0.000, 0.000, 0.153, 0.161, 0.000],
    [0.000, 0.191, 0.000, 0.000, 0.225, 0.029, 0.000]],
    "thermogram_30.npy":
    [[0.000, 0.103, 0.000, 0.000, 0.156, 0.176, 0.000],
    [0.000, 0.055, 0.000, 0.000, 0.119, 0.031, 0.000],
    [0.000, 0.125, 0.000, 0.000, 0.136, 0.125, 0.000],
    [0.000, 0.103, 0.000, 0.000, 0.109, 0.219, 0.000],
    ],
    "thermogram_31.npy":
    [[0.000, 0.076, 0.000, 0.000, 0.128, 0.161, 0.000],
    [0.000, 0.040, 0.000, 0.000, 0.118, 0.037, 0.000],
    [0.000, 0.076, 0.000, 0.000, 0.108, 0.055, 0.000],
    [0.000, 0.075, 0.000, 0.000, 0.055, 0.162, 0.000],
    ],
    "thermogram_32.npy":
    [[0.000, 0.075, 0.000, 0.000, 0.073, 0.148, 0.000],
    [0.000, 0.044, 0.000, 0.000, 0.109, 0.000, 0.000],
    [0.000, 0.200, 0.000, 0.000, 0.195, 0.117, 0.000],
    [0.000, 0.164, 0.000, 0.000, 0.163, 0.022, 0.000],
    ],
    "thermogram_33.npy":
    [[0.000, 0.063, 0.000, 0.081, 0.153, 0.218, 0.000],
    [0.000, 0.117, 0.000, 0.000, 0.189, 0.175, 0.000],
    [0.000, 0.088, 0.000, 0.000, 0.150, 0.066, 0.000],
    [0.000, 0.071, 0.000, 0.000, 0.147, 0.123, 0.000],
    ],
    "thermogram_34.npy":
    [[0.000, 0.061, 0.080, 0.000, 0.149, 0.222, 0.000],
    [0.000, 0.098, 0.000, 0.082, 0.000, 0.262, 0.000],
    [0.000, 0.106, 0.000, 0.094, 0.000, 0.250, 0.000],
    [0.000, 0.068, 0.000, 0.000, 0.000, 0.000, 0.000]
    ],
    "thermogram_36.npy":
    [[0.000, 0.068, 0.000, 0.000, 0.190, 0.184, 0.000],
    [0.000, 0.105, 0.000, 0.000, 0.117, 0.065, 0.000],
    [0.000, 0.044, 0.000, 0.000, 0.229, 0.028, 0.000],
    [0.000, 0.101, 0.000, 0.000, 0.254, 0.092, 0.000]
    ],
    "thermogram_38.npy":
    [[0.000, 0.000, 0.000, 0.287, 0.155, 0.040, 0.000],
    [0.000, 0.050, 0.000, 0.000, 0.133, 0.077, 0.000],
    [0.000, 0.106, 0.000, 0.000, 0.183, 0.012, 0.000],
    [0.000, 0.087, 0.000, 0.000, 0.098, 0.045, 0.000]
    ],
    "thermogram_39.npy":
    [[0.000, 0.093, 0.000, 0.000, 0.129, 0.033, 0.000],
    [0.000, 0.137, 0.000, 0.000, 0.074, 0.033, 0.000],
    [0.000, 0.095, 0.000, 0.000, 0.204, 0.000, 0.000],
    [0.000, 0.146, 0.000, 0.000, 0.207, 0.066, 0.000]
    ],
    "thermogram_40.npy":
    [[0.000, 0.000, 0.000, 0.000, 0.146, 0.063, 0.293],
    [0.000, 0.000, 0.000, 0.000, 0.128, 0.017, 0.361],
    [0.000, 0.000, 0.000, 0.000, 0.054, 0.029, 0.000],
    [0.131, 0.070, 0.000, 0.000, 0.110, 0.082, 0.000]
    ],
    "thermogram_41.npy":
    [[0.000, 0.124, 0.000, 0.000, 0.074, 0.038, 0.000],
    [0.000, 0.077, 0.000, 0.000, 0.092, 0.035, 0.000],
    [0.000, 0.084, 0.000, 0.000, 0.164, 0.015, 0.000],
    [0.000, 0.111, 0.000, 0.000, 0.123, 0.111, 0.000]
    ],
    "thermogram_42.npy":
    [[0.000, 0.250, 0.000, 0.000, 0.221, 0.032, 0.000],
    [0.000, 0.132, 0.000, 0.000, 0.157, 0.022, 0.000],
    [0.000, 0.171, 0.000, 0.000, 0.192, 0.032, 0.000],
    [0.000, 0.053, 0.000, 0.000, 0.066, 0.128, 0.000]
    ],
    "thermogram_43.npy":
    [[0.000, 0.106, 0.000, 0.000, 0.102, 0.097, 0.000],
    [0.000, 0.100, 0.000, 0.000, 0.200, 0.200, 0.000],
    [0.000, 0.113, 0.000, 0.000, 0.306, 0.000, 0.000],
    [0.000, 0.134, 0.000, 0.000, 0.190, 0.127, 0.000]
    ],
    }


class Conv3DDataset(Dataset):
    def __init__(self, video_path: str, labels: List[List[int]], input_shape: Tuple[int, int], 
                 w_size: int, overlap: float = 0, offset: int = 0):
        super().__init__()
        self.w_size = w_size
        self.data = np.load(video_path, mmap_mode='r')  # load npy per class in mmap mode
        self.t_min = self.data.min()
        self.t_max = self.data.max()
        self.labels = torch.tensor(labels)
        self.input_shape = input_shape
        self.offset = offset
        n_frames = self.data.shape[0] - 2 * offset  # total number of frames to be used
        self.step = floor((1 - overlap) * w_size)  # step of the sliding window
        self.n_steps = (n_frames - w_size) // self.step # number of steps for video
        self.n_frames = n_frames

    def __len__(self) -> int:
        return self.n_steps
    
    def resize(self, data: np.ndarray) -> np.ndarray:
        x_scale = self.input_shape[0] / data.shape[2]
        y_scale = self.input_shape[1] / data.shape[1]
        return ndimage.zoom(data, (1, y_scale, x_scale))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n = idx * self.step + self.offset
        frames = self.data[n : n + self.w_size].copy()
        frames -= self.t_min
        frames = frames / (self.t_max - self.t_min)
        frames = self.resize(frames)
        frames = torch.from_numpy(frames)
        cl = self.labels[int(n / self.n_frames * self.labels.shape[0])]
        return torch.permute(frames[..., None].float(), (3, 0, 1, 2)), cl.float()
        # return frames.float(), cl.float()


class Conv2DDataset(Dataset):
    def __init__(self, video_path: str, labels: List[List[int]], input_shape: Tuple[int, int]):
        super().__init__()
        self.data = np.load(video_path, mmap_mode='r')  # load npy per class in mmap mode
        self.t_min = self.data.min()
        self.t_max = self.data.max()
        self.labels = torch.tensor(labels)
        self.input_shape = input_shape
        n_frames = self.data.shape[0] # total number of frames to be used
        self.n_frames = n_frames

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frames = self.data[idx].copy()
        frames -= self.t_min
        frames = frames / (self.t_max - self.t_min)
        frames = cv2.resize(frames, self.input_shape)
        frames = torch.from_numpy(frames)
        cl = self.labels[int(idx / self.n_frames * self.labels.shape[0])]
        return frames.float().unsqueeze(0), cl.float()
    

def make_datasets(markup_path: str, npy_dir: str, classes: List[str], w_size=8, overlap=0.1, offset=10, val_videos=None) -> Tuple[ConcatDataset, ConcatDataset]:
    with open(markup_path, 'r') as f:
        markup = json.load(f)
    videos = list(markup.keys())
    labels = {}
    for vid in videos:
        label = 0
        for d in markup[vid]:
            if d[-1] > 0:
                label = 1

        labels[vid] = label

    videos = list(labels.keys())
    if val_videos is None:
        train_videos, val_videos = train_test_split(videos, test_size=0.3, random_state=42, stratify=list(labels.values()))
    else:
        train_videos = [vid for vid in videos if vid not in val_videos]
    print(val_videos)
    all_classes = ('hu', 'hg', 'he','hp', 'hs', 'hm', 'hi')
    idx = [all_classes.index(cl) for cl in classes]
    train_dss = [Conv3DDataset(f"{npy_dir}/{vid}", np.array(markup[vid])[:, idx].astype(bool).astype(int).tolist(), (320, 240), 
                               w_size=w_size,
                               overlap=overlap,
                               offset=offset) for vid in train_videos]
    val_dss = [Conv3DDataset(f"{npy_dir}/{vid}", np.array(markup[vid])[:, idx].astype(bool).astype(int).tolist(), (320, 240), 
                               w_size=w_size,
                               overlap=overlap,
                               offset=offset) for vid in val_videos]
    return ConcatDataset(train_dss), ConcatDataset(val_dss)


def make_conv2d_datasets(markup_path: str, npy_dir: str, classes: List[str], val_videos=None) -> Tuple[ConcatDataset, ConcatDataset]:
    with open(markup_path, 'r') as f:
        markup = json.load(f)
    videos = list(markup.keys())
    labels = {}
    for vid in videos:
        label = 0
        for d in markup[vid]:
            if d[-1] > 0:
                label = 1

        labels[vid] = label

    videos = list(labels.keys())
    if val_videos is None:
        train_videos, val_videos = train_test_split(videos, test_size=0.3, random_state=42, stratify=list(labels.values()))
    else:
        train_videos = [vid for vid in videos if vid not in val_videos]
    print(val_videos)
    all_classes = ('hu', 'hg', 'he','hp', 'hs', 'hm', 'hi')
    idx = [all_classes.index(cl) for cl in classes]
    train_dss = [Conv2DDataset(f"{npy_dir}/{vid}", np.array(markup[vid])[:, idx].astype(bool).astype(int).tolist(), (320, 240)) for vid in train_videos]
    val_dss = [Conv2DDataset(f"{npy_dir}/{vid}", np.array(markup[vid])[:, idx].astype(bool).astype(int).tolist(), (320, 240)) for vid in val_videos]
    return ConcatDataset(train_dss), ConcatDataset(val_dss)


def prepare_spatters_dataset(json_path: str, val_videos: List[str], train_videos: List[str], classes: List[str],
                      window_size: int = 8, overlap_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    def trim_zeros_and_process(video_data: Dict) -> Dict:
        """Пункт 1: Удаление нулей с краев welding_zone_temp"""
        welding_temp = np.array(video_data['welding_zone_temp'])
        
        # Находим индексы ненулевых элементов
        non_zero_indices = np.where(welding_temp != 0)[0]
        
        if len(non_zero_indices) == 0:
            # Если все значения нулевые, возвращаем пустые массивы
            return {key: np.array([]) for key in video_data.keys()}
        
        start_idx = non_zero_indices[0]
        end_idx = non_zero_indices[-1] + 1
        
        # Обрезаем все массивы
        trimmed_data = {}
        for key, values in video_data.items():
            trimmed_data[key] = np.array(values)[start_idx:end_idx]
            
        return trimmed_data
    
    def split_into_quarters(data: np.ndarray) -> List[np.ndarray]:
        """Пункт 2: Разделение на 4 равные части"""
        if len(data) == 0:
            return [np.array([]) for _ in range(4)]
        
        part_length = len(data) // 4
        parts = []
        
        for i in range(4):
            start_idx = i * part_length
            if i == 3:  # Последняя часть - все оставшиеся элементы
                end_idx = len(data)
            else:
                end_idx = (i + 1) * part_length
            parts.append(data[start_idx:end_idx])
            
        return parts
    
    def calculate_moving_stats(data: np.ndarray, window_size: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
        """Вычисление скользящего среднего и СКО с заданным шагом"""
        if len(data) < window_size:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
        means = []
        stds = []
        medians = []
        rms = []  # среднеквадратичное значение
        slopes = []  # наклон тренда
        
        for i in range(0, len(data) - window_size + 1, step):
            window = data[i:i + window_size]
            means.append(np.mean(window))
            stds.append(np.std(window))
        return np.array(means), np.array(stds)
    
    def calculate_step_size(window_size: int, overlap_ratio: float) -> int:
        """Вычисление шага для скользящего окна на основе доли пересечения"""
        if overlap_ratio <= 0:
            return window_size  # Без пересечения
        elif overlap_ratio >= 1:
            return 1  # Максимальное пересечение (шаг 1)
        else:
            # Вычисляем шаг: window_size * (1 - overlap_ratio)
            step = int(window_size * (1 - overlap_ratio))
            return max(1, step)  # Шаг не меньше 1
        
    def contains_zeros(window_data: np.ndarray) -> bool:
        """Проверяет, содержит ли окно нулевые значения"""
        return np.any(window_data == 0)
    
    processed_data = {}
    
    # Вычисляем шаг для окон
    step_size = calculate_step_size(window_size, overlap_ratio)
    
    for video_name, video_data in json_data.items():
        # 1. Удаляем нули с краев
        trimmed_data = trim_zeros_and_process(video_data)
        
        if len(trimmed_data['welding_zone_temp']) == 0:
            continue
            
        # 2. Разделяем на 4 части и получаем GT значения для каждой четверти
        quarters_data = {}
        
        # Разделяем все признаки на четверти
        for key in ['size', 'n_spatters', 'temp', 'welding_zone_temp']:
            quarters = split_into_quarters(trimmed_data[key])
            quarters_data[key] = quarters
        
        # Получаем GT значения для каждой четверти (предполагаем, что они уже есть в данных)
        # Если GT передаются отдельно, нужно изменить эту часть
        # gt_quarters = split_into_quarters(trimmed_data['welding_zone_temp'])
        # quarter_gt_values = [np.mean(q) if len(q) > 0 else 0 for q in gt_quarters]
        
        # 3. Вычисляем скользящие статистики для каждой четверти
        features = []
        gt_for_features = []
        
        for quarter_idx in range(4):
            # Получаем данные для текущей четверти
            quarter_data_size = quarters_data['size'][quarter_idx]
            quarter_data_spatters = quarters_data['n_spatters'][quarter_idx]
            quarter_data_temp = quarters_data['temp'][quarter_idx]
            quarter_data_welding = quarters_data['welding_zone_temp'][quarter_idx]
            
            # GT для всей этой четверти
            quarter_gt = QUALITY[video_name][quarter_idx]
            
            if len(quarter_data_size) >= window_size:
                size_means, size_stds = calculate_moving_stats(quarter_data_size, window_size, step_size)
                spatter_means, spatter_stds = calculate_moving_stats(quarter_data_spatters, window_size, step_size)
                temp_means, temp_stds = calculate_moving_stats(quarter_data_temp, window_size, step_size)
                welding_means, welding_stds = calculate_moving_stats(quarter_data_welding, window_size, step_size)

                # Для каждого окна создаем вектор признаков
                for i in range(len(size_means)):
                    window_features = {
                        # Основные статистики (работали хорошо)
                        'size_mean': size_means[i],
                        'size_std': size_stds[i],
                        'n_spatters_mean': spatter_means[i],
                        'n_spatters_std': spatter_stds[i],
                        'temp_mean': temp_means[i],
                        'temp_std': temp_stds[i],
                        'welding_zone_temp_mean': welding_means[i],
                        'welding_zone_temp_std': welding_stds[i],
                    }
                    
                    features.append(window_features)
                    gt_for_features.append(quarter_gt)  # Всем окнам из этой четверти - одинаковый GT
        
        processed_data[video_name] = {
            'features': features,
            'gt': gt_for_features
        }
    
    # 4. Создаем датасеты
    def create_dataset(video_names: List[str]) -> pd.DataFrame:
        """Создание датасета для списка видео"""
        rows = []
        
        for video_name in video_names:
            if video_name not in processed_data:
                continue
                
            video_info = processed_data[video_name]
            
            # Создаем строки для каждого окна
            for i, feature_set in enumerate(video_info['features']):
                row = feature_set.copy()
                gt = video_info['gt'][i]
                for i, name in enumerate(('hu', 'hg', 'he','hp', 'hs', 'hm', 'hi')):
                    if name in classes:
                        row[name] = gt[i]  # GT для этого окна (одинаковый для всей четверти)
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    # Создаем тренировочный и валидационный датасеты
    train_df = create_dataset(train_videos)
    val_df = create_dataset(val_videos)
    y_train = train_df[list(classes)]
    X_train = train_df.drop(list(classes), axis=1)
    y_val = val_df[list(classes)]
    X_val = val_df.drop(list(classes), axis=1)
    return X_train, y_train, X_val, y_val


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_nn_mlb_clf(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray, name: str) -> Tuple[float, np.ndarray]:
    n_features = X_train.shape[1]
    train_ds = torch.cat((torch.from_numpy(X_train), torch.from_numpy(y_train)), dim=1)  # creat dataset
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)  # create dataloader
    val_ds = torch.from_numpy(X_val)  # create dataset
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)  # create dataloader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    train_loss = nn.BCEWithLogitsLoss()
    best_f1 = 0
    for _ in range(100):
        model.train()
        for batch in train_dl:
            X_train, y_train = batch[:, :n_features].to(device).float(), batch[:, n_features:].to(device).float()
            y_pred = model(X_train)
            loss = train_loss(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()  # back propogation
            optimizer.step()  # optimizer's step

        model.eval()
        total_preds = []
        for batch in val_dl:
            batch = batch.to(device).float()
            with torch.no_grad():
                y_pred = model(batch)
            total_preds.append(y_pred.sigmoid().cpu().numpy())
        y_pred = np.concatenate(total_preds)
        y_pred = (y_pred > 0.5).astype(int)
        # print(y_pred)
        # print(y_pred)
        pr, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), name)

    model.load_state_dict(torch.load(name))
    total_preds = []
    for batch in val_dl:
        batch = batch.to(device).float()
        with torch.no_grad():
            y_pred = model(batch)
        total_preds.append(y_pred.sigmoid().cpu().numpy())
    y_pred = np.concatenate(total_preds)

    return best_f1, y_pred


def prepare_sequences_dataset(json_path: str, val_videos: List[str], train_videos: List[str], classes: List[str],
                      sequence_length: int = 32, overlap_ratio: float = 0.5) -> Tuple[Dict[str, List], Dict[str, List]]:
    """
    Подготавливает dataset из последовательностей фиксированной длины с перекрытием
    
    Returns:
        train_sequences: dict с ключами 'sequences' и 'labels'
        val_sequences: dict с ключами 'sequences' и 'labels'
    """

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    def trim_zeros_and_process(video_data: Dict) -> Dict:
        """Пункт 1: Удаление нулей с краев welding_zone_temp"""
        welding_temp = np.array(video_data['welding_zone_temp'])
        
        # Находим индексы ненулевых элементов
        non_zero_indices = np.where(welding_temp != 0)[0]
        
        if len(non_zero_indices) == 0:
            return {key: np.array([]) for key in video_data.keys()}
        
        start_idx = non_zero_indices[0]
        end_idx = non_zero_indices[-1] + 1
        
        # Обрезаем все массивы
        trimmed_data = {}
        for key, values in video_data.items():
            trimmed_data[key] = np.array(values)[start_idx:end_idx]
            
        return trimmed_data
    
    def split_into_quarters(data: np.ndarray) -> List[np.ndarray]:
        """Пункт 2: Разделение на 4 равные части"""
        if len(data) == 0:
            return [np.array([]) for _ in range(4)]
        
        part_length = len(data) // 4
        parts = []
        
        for i in range(4):
            start_idx = i * part_length
            if i == 3:
                end_idx = len(data)
            else:
                end_idx = (i + 1) * part_length
            parts.append(data[start_idx:end_idx])
            
        return parts
    
    def calculate_step_size(sequence_length: int, overlap_ratio: float) -> int:
        """Вычисление шага для скользящего окна на основе доли пересечения"""
        if overlap_ratio <= 0:
            return sequence_length
        elif overlap_ratio >= 1:
            return 1
        else:
            step = int(sequence_length * (1 - overlap_ratio))
            return max(1, step)
    
    def create_sequences_from_quarter(quarter_data: Dict[str, np.ndarray], quarter_gt: List, 
                                    seq_length: int, step: int) -> Tuple[List[np.ndarray], List[List]]:
        """Создает последовательности фиксированной длины из данных четверти"""
        # Проверяем, что все массивы одинаковой длины
        lengths = [len(quarter_data[key]) for key in ['size', 'n_spatters', 'temp', 'welding_zone_temp']]
        if len(set(lengths)) != 1:
            min_len = min(lengths)
            for key in quarter_data:
                quarter_data[key] = quarter_data[key][:min_len]
        
        total_length = len(quarter_data['size'])
        if total_length < seq_length:
            return [], []
        
        sequences = []
        labels = []
        
        # Создаем последовательности с заданным шагом
        for start_idx in range(0, total_length - seq_length + 1, step):
            end_idx = start_idx + seq_length
            
            # Создаем многомерный временной ряд: (seq_length, 4_features)
            sequence = np.column_stack([
                quarter_data['size'][start_idx:end_idx],
                quarter_data['n_spatters'][start_idx:end_idx],
                quarter_data['temp'][start_idx:end_idx],
                quarter_data['welding_zone_temp'][start_idx:end_idx]
            ])
            
            sequences.append(sequence)
            labels.append(quarter_gt)  # Та же метка для всех последовательностей из этой четверти
        
        return sequences, labels
    
    processed_sequences = {}
    step_size = calculate_step_size(sequence_length, overlap_ratio)
    
    for video_name, video_data in json_data.items():
        # 1. Удаляем нули с краев
        trimmed_data = trim_zeros_and_process(video_data)
        
        if len(trimmed_data['welding_zone_temp']) == 0:
            continue
            
        # 2. Разделяем на 4 части
        quarters_data = {}
        for key in ['size', 'n_spatters', 'temp', 'welding_zone_temp']:
            quarters = split_into_quarters(trimmed_data[key])
            quarters_data[key] = quarters
        
        video_sequences = []
        video_labels = []
        
        for quarter_idx in range(4):
            # Собираем данные для текущей четверти
            quarter_data = {
                'size': quarters_data['size'][quarter_idx],
                'n_spatters': quarters_data['n_spatters'][quarter_idx],
                'temp': quarters_data['temp'][quarter_idx],
                'welding_zone_temp': quarters_data['welding_zone_temp'][quarter_idx]
            }
            
            # GT для этой четверти
            quarter_gt = QUALITY[video_name][quarter_idx]
            quarter_gt = [quarter_gt[i] for i, name in enumerate(['hu', 'hg', 'he', 'hp', 'hs', 'hm', 'hi']) 
               if name in classes]
            
            # Создаем последовательности фиксированной длины
            sequences, labels = create_sequences_from_quarter(
                quarter_data, quarter_gt, sequence_length, step_size
            )
            
            video_sequences.extend(sequences)
            video_labels.extend(labels)
        
        processed_sequences[video_name] = {
            'sequences': video_sequences,
            'labels': video_labels
        }
    
    def create_sequences_dataset(video_names: List[str]) -> Dict[str, List]:
        """Создание dataset из последовательностей для списка видео"""
        all_sequences = []
        all_labels = []
        all_video_names = []
        all_quarter_indices = []
        
        for video_name in video_names:
            if video_name not in processed_sequences:
                continue
                
            video_info = processed_sequences[video_name]
            
            for sequence, label in zip(video_info['sequences'], video_info['labels']):
                all_sequences.append(sequence)
                all_labels.append(label)
                all_video_names.append(video_name)
        
        # return {
        #     'sequences': all_sequences,  # list of numpy arrays shape (sequence_length, 4)
        #     'labels': all_labels,        # list of GT lists
        #     'video_names': all_video_names
        # }
        return all_sequences, all_labels
    
    # Создаем тренировочный и валидационный datasets
    X_train, y_train = create_sequences_dataset(train_videos)
    X_val, y_val = create_sequences_dataset(val_videos)
    
    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)


def train_conv1d_mlb_clf(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray, name: str) -> Tuple[float, np.ndarray]:
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))  # creat dataset
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)  # create dataloader
    val_ds = torch.from_numpy(X_val)  # create dataset
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)  # create dataloader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    train_loss = nn.BCEWithLogitsLoss()
    # train_loss = AsymmetricLoss()
    best_f1 = 0
    for _ in range(100):
        model.train()
        for X_train, y_train in train_dl:
            X_train, y_train = X_train.to(device).float(), y_train.to(device).float()
            y_pred = model(X_train)
            loss = train_loss(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()  # back propogation
            optimizer.step()  # optimizer's step

        model.eval()
        total_preds = []
        for batch in val_dl:
            batch = batch.to(device).float()
            with torch.no_grad():
                y_pred = model(batch)
            total_preds.append(y_pred.sigmoid().cpu().numpy())
        y_pred = np.concatenate(total_preds)
        y_pred = (y_pred > 0.5).astype(int)
        # print(y_pred)
        # print(y_pred)
        pr, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), name)

    model.load_state_dict(torch.load(name))
    total_preds = []
    for batch in val_dl:
        batch = batch.to(device).float()
        with torch.no_grad():
            y_pred = model(batch)
        total_preds.append(y_pred.sigmoid().cpu().numpy())
    y_pred = np.concatenate(total_preds)

    return best_f1, y_pred


class AsymmetricLoss(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()
