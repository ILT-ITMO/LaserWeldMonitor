import struct
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple
from glob import glob
from math import floor
from sklearn.model_selection import train_test_split


def process_raw_data(path):
    with open(path, 'rb') as f:
        print(f.read(4))  # useless pre-header
        header, = struct.unpack('167s', f.read(167))
        header = header.decode()
        header = json.loads(header)
        # print(header)
        size = header['BasicInfo']['length']  # from header, totalling 8 blocks, 4 channels
        # f.read(5)
        data = f.read(size * 8)
        d = struct.unpack(f'{size * 8 // 2}h', data)
        # print(len(d))
    sig = np.array(d).reshape(4, -1) / (3.2*10e2)
    # print(sig.shape)
    id = header['BasicInfo']['datetime'].split('-')[-1]
    np.save(f'photodiode_analysis/data/{id}.npy', sig)


def visualize_signal(path: str):
    data = np.load('signals/73.npy')
    print(data.shape)
    titles = ["Visible Sensor Signal", "Reflected Sensor Signal", "Infrared Sensor Signal"]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 10), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    for i in range(3):
        axes[i].plot(data[1 + i,:])
        axes[i].set_title(titles[i])
        axes[i].set_ylabel("Voltage, V")
        axes[i].set_ylim((0, 10))
        axes[i].grid(True)
    plt.ylim((0, 10))
    plt.xlim((10000, 50000))
    plt.xlabel("Time")
    plt.show()


def load_gt(path: str) -> pd.DataFrame:
    with open(path, 'r') as f:
        gt = json.load(f)
    
    column_names = ['hu', 'hg', 'he', 'hp', 'hs', 'hm', 'hi']
    df = pd.DataFrame.from_dict(gt, 'index')
    df.columns = column_names
    return df


def process_signal_mean_std(signal: np.ndarray, w_size: int, area_size: int, overlap: float = 0.2) -> np.ndarray:
    signal = signal[1:, :]  # remove frequency
    reflected = signal[1]
    mask = reflected > 0.1
    signal = signal[:, mask]
    center_id = signal.shape[1] // 2
    signal = signal[:, center_id - area_size // 2 : center_id + area_size // 2]  # area to process
    step = floor((1 - overlap) * w_size)
    n_steps = (area_size - w_size) // step + 1
    inter = np.stack([signal[:, i * step : i * step + w_size] for i in range(n_steps)], axis=0)
    return np.stack((inter.mean(axis=-1), inter.std(axis=-1)), axis=-1)


def generate_ds_mean_std(src_dir: str, gt_path: str, area_size: int, w_size: int,
                         overlap: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(glob(f"{src_dir}/*.npy"))
    df = np.stack([process_signal_mean_std(np.load(file), w_size, area_size, overlap).reshape(-1, 6) for file in files], axis=0)
    n_steps = df.shape[1]
    # print(df.shape)
    df = pd.DataFrame(df.reshape(-1, 6))
    df.columns = ['mean_v', 'sigma_v', 'mean_r', 'sigma_r', 'mean_i', 'sigma_i']
    gt_df = load_gt(gt_path)
    gt = gt_df.to_numpy().tolist()
    gt = np.concatenate([np.array(g * n_steps).reshape(n_steps, -1) for g in gt], axis=0)
    gt = pd.DataFrame(gt)
    gt.columns = gt_df.columns
    return df, gt


def generate_ds_split_mean_std(src_dir: str, gt_path: str, area_size: int, w_size: int,
                         overlap: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    ids = list(gt.keys())
    train_id, val_id = train_test_split(ids, test_size=0.3)
    train_gt = [gt[id] for id in train_id]
    val_gt = [gt[id] for id in val_id]
    train_sig = [f"{src_dir}/{id}.npy" for id in train_id]
    val_sig = [f"{src_dir}/{id}.npy" for id in val_id]
    train_feat = np.stack([process_signal_mean_std(np.load(file), w_size, area_size, overlap).reshape(-1, 6) for file in train_sig], axis=0)
    val_feat = np.stack([process_signal_mean_std(np.load(file), w_size, area_size, overlap).reshape(-1, 6) for file in val_sig], axis=0)
    n_steps = train_feat.shape[1]
    # print(df.shape)
    # Train
    train_feat = pd.DataFrame(train_feat.reshape(-1, 6))
    train_feat.columns = ['mean_v', 'sigma_v', 'mean_r', 'sigma_r', 'mean_i', 'sigma_i']
    train_gt = np.concatenate([np.array(g * n_steps).reshape(n_steps, -1) for g in train_gt], axis=0)
    train_gt = pd.DataFrame(train_gt)
    train_gt.columns = ['hu', 'hg', 'he', 'hp', 'hs', 'hm', 'hi']
    # Val
    val_feat = pd.DataFrame(val_feat.reshape(-1, 6))
    val_feat.columns = ['mean_v', 'sigma_v', 'mean_r', 'sigma_r', 'mean_i', 'sigma_i']
    val_gt = np.concatenate([np.array(g * n_steps).reshape(n_steps, -1) for g in val_gt], axis=0)
    val_gt = pd.DataFrame(val_gt)
    val_gt.columns = ['hu', 'hg', 'he', 'hp', 'hs', 'hm', 'hi']
    return train_feat, train_gt, val_feat, val_gt
