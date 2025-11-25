import numpy as np
import json
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_recall_fscore_support
from utils import QUALITY, prepare_spatters_dataset, seed_everything, train_nn_mlb_clf, prepare_sequences_dataset, train_conv1d_mlb_clf
from sklearn.model_selection import StratifiedKFold
from models import NNClassifier, TabularMLP, CNNTimeSeriesClassifier, HybridCNNTransformerMultilabelV2, MultiLayerBiLSTMClassifier, TransformerClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from torchsummary import summary


seed_everything(42)
videos = list(QUALITY.keys())

with open('without_tracker.json', 'r') as f:
    data = json.load(f)

videos = list(data.keys())

labels = {}
gt = {vid: [] for vid in videos}
cl = 6  # 1 - отл, 3 - не оч, 4 - отл, 5 - отл, 6 - хор
for vid in videos:
    label = 0
    for d in QUALITY[vid]:
        if d[6] > 0:
            label = 1
        gt[vid].append(int(d[cl] > 0))

    labels[vid] = label


folds = 4
classes = ('hg', 'hs', 'hm', 'hi')
k_fold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
output = {}


HIDDEN_LAYERS = 512
# WINDOW_SIZES = [8, 16, 32, 64, 96, 128]
WINDOW_SIZES = [128]

total_map = []

for ws in WINDOW_SIZES:
    y_real = []
    y_proba = []
    f1_folds = []
    map_folds = []

    for i, (train_index, val_index) in tqdm(enumerate(k_fold.split(list(labels.keys()), list(labels.values()))), total=folds):
        train_videos = [list(labels.keys())[v] for v in train_index]
        val_videos = [list(labels.keys())[v] for v in val_index]
        # print(val_videos)

        # X_train, y_train, X_val, y_val = prepare_spatters_dataset('without_tracker.json', val_videos, train_videos, 
        #                                                           classes, window_size=ws, overlap_ratio=0.5)
        # X_train = X_train.to_numpy()
        # y_train = y_train.to_numpy()
        # X_val = X_val.to_numpy()
        # y_val = y_val.to_numpy()
        X_train, y_train, X_val, y_val = prepare_sequences_dataset('without_tracker.json', val_videos, train_videos, 
                                                                classes, sequence_length=ws, overlap_ratio=0.5)
        # model = CNNTimeSeriesClassifier(X_train.shape[2], X_train.shape[1], len(classes), conv_filters=[64, 128, 256], kernel_sizes=[7, 5, 3],
        #             dropout_rate=0.3, use_batch_norm=True)
        model = HybridCNNTransformerMultilabelV2(X_train.shape[2], X_train.shape[1], len(classes))
        # model = MultiLayerBiLSTMClassifier(X_train.shape[2], 128, 4, len(classes))
        # model = TransformerClassifier(X_train.shape[2], X_train.shape[1], len(classes), d_model=256, nhead=8, num_layers=3, 
        #              dim_feedforward=512, dropout=0.1)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {trainable_params}")
        y_val = (y_val > 0).astype(int)
        y_train = (y_train > 0).astype(int)
        name = f"fold_{i}_best_f1.pt"
        best_f1, y_pred = train_conv1d_mlb_clf(model, X_train, y_train, X_val, y_val, name)
        # model = TabularMLP(X_train.shape[1], len(classes), hidden_dim=HIDDEN_LAYERS, dropout_rate=0.1)
        
        # best_f1, y_pred = train_nn_mlb_clf(model, X_train, y_train, X_val, y_val, name)
        mAP = 0
        for cl_id, cl in enumerate(classes):
            mAP += average_precision_score(y_val[:, cl_id], y_pred[:, cl_id])
        
        map_folds.append(mAP / len(classes))
        y_real.append(y_val)
        y_proba.append(y_pred)

    map_folds = np.array(map_folds)
    print(f"Fold-wise mAP for window {ws}: {map_folds.mean():.3f} +- {map_folds.std():.3f}")
    total_map.append((map_folds.mean(), map_folds.std()))


# print(f"Model summary for hidden dim {HIDDEN_LAYERS}")


for ws, (m, std) in zip(WINDOW_SIZES, total_map):
    print(f"Fold-wise mAP for window {ws}: {m:.3f} +- {std:.3f}")