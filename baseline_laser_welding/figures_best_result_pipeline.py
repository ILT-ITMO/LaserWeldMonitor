import numpy as np
import json
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_recall_fscore_support
from utils import QUALITY, prepare_sequences_dataset, seed_everything, train_conv1d_mlb_clf, prepare_spatters_dataset, train_nn_mlb_clf
from sklearn.model_selection import StratifiedKFold
from models import HybridCNNTransformerMultilabelV2, TabularMLP
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
WINDOW_SIZE = 64
THRESHOLD = 0.55

y_real = []
y_proba = []
f1_folds = []
prec_folds = []
rec_folds = []
ap_folds = []

for i, (train_index, val_index) in tqdm(enumerate(k_fold.split(list(labels.keys()), list(labels.values()))), total=folds):
    aps = []
    train_videos = [list(labels.keys())[v] for v in train_index]
    val_videos = [list(labels.keys())[v] for v in val_index]
    X_train, y_train, X_val, y_val = prepare_sequences_dataset('without_tracker.json', val_videos, train_videos, 
                                                                classes, sequence_length=WINDOW_SIZE, overlap_ratio=0.5)
    # X_train, y_train, X_val, y_val = prepare_spatters_dataset('without_tracker.json', val_videos, train_videos, 
    #                                                           classes, window_size=96, overlap_ratio=0.5)
    # X_train = X_train.to_numpy()
    # y_train = y_train.to_numpy()
    # X_val = X_val.to_numpy()
    # y_val = y_val.to_numpy()
    y_val = (y_val > 0).astype(int)
    y_train = (y_train > 0).astype(int)
    # model = TabularMLP(X_train.shape[1], len(classes), hidden_dim=512, dropout_rate=0.1)
    model = HybridCNNTransformerMultilabelV2(X_train.shape[2], X_train.shape[1], len(classes))
    name = f"fold_{i}_best_f1.pt"
    best_f1, y_pred = train_conv1d_mlb_clf(model, X_train, y_train, X_val, y_val, name)
    # best_f1, y_pred = train_nn_mlb_clf(model, X_train, y_train, X_val, y_val, name)
    # print(f"Best F1-score for fold {i}: {best_f1:.3f}")
    pr, rec, f1, _ = precision_recall_fscore_support(y_val, (y_pred > THRESHOLD).astype(int), average=None)
    f1_folds.append(f1)
    prec_folds.append(pr)
    rec_folds.append(rec)
    for cl_id, cl in enumerate(classes):
        ap = average_precision_score(y_val[:, cl_id], y_pred[:, cl_id])
        aps.append(ap)
    ap_folds.append(aps)
    y_real.append(y_val)
    y_proba.append(y_pred)

y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
f1_folds = np.array(f1_folds)
prec_folds = np.array(prec_folds)
rec_folds = np.array(rec_folds)
ap_folds = np.array(ap_folds)
print(f"Precision: {prec_folds.mean(axis=0)} +- {prec_folds.std(axis=0)} {prec_folds.mean()}")
print(f"Recall: {rec_folds.mean(axis=0)} +- {rec_folds.std(axis=0)} {rec_folds.mean()}")
print(f"F1-score: {f1_folds.mean(axis=0)} +- {f1_folds.std(axis=0)} {f1_folds.mean()}")
print(f"AP: {ap_folds.mean(axis=0)} +- {ap_folds.std(axis=0)} {ap_folds.mean()}")
# pr, rec, f1, _ = precision_recall_fscore_support(y_real, (y_proba > THRESHOLD).astype(int), average=None)
# print(f"Precision: {pr} macro {pr.mean()}")
# print(f"Recall: {rec} macro {rec.mean()}")
# print(f"F1-score: {f1} macro {f1.mean()}")

ap_sum = 0

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

for cl_id, cl in enumerate(classes):
    pr, rec, _ = precision_recall_curve(y_real[:, cl_id], y_proba[:, cl_id])
    ap = average_precision_score(y_real[:, cl_id], y_proba[:, cl_id])
    ap_sum += ap
    label = f"{cl} AP={ap:.3f}"
    axes[0].plot(rec, pr, label=label)

pr, rec, _ = precision_recall_curve(y_real.flatten(), y_proba.flatten())
label = f"Total mAP={round(ap_sum / len(classes), 3)}"
axes[0].plot(rec, pr, linewidth=3, label=label, color='black')
axes[0].set_xlabel('Recall')
axes[0].set_ylabel('Precision')
axes[0].legend()
axes[0].grid()

axes[0].set_title('a')
axes[0].axis('equal')
axes[0].set(xlim=(0, 1), ylim=(0, 1))


thresholds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
out = {'thresholds': thresholds, 'precision': [], 'recall': [], 'f1-score': []}
for th in thresholds:
    pr, rec, f1, _ = precision_recall_fscore_support(y_real, (y_proba > th) * 1, average='macro')
    out['precision'].append(pr)
    out['recall'].append(rec)
    out['f1-score'].append(f1)

out['thresholds'].append(1)
out['precision'].append(1)
out['recall'].append(0)
out['f1-score'].append(0)
th = pd.DataFrame.from_dict(out)
axes[1].plot(th['thresholds'], th['precision'], label='Precision', lw=3)
axes[1].plot(th['thresholds'], th['recall'], label='Recall', lw=3)
axes[1].plot(th['thresholds'], th['f1-score'], label='F1-score', lw=3)
axes[1].set_xlabel('Threshold')
axes[1].set_ylabel('Metric')
axes[1].legend()
axes[1].grid()
axes[1].set_title('b')
axes[1].axis('equal')
axes[1].set(xlim=(0, 1), ylim=(0, 1))

plt.savefig('figures/PR_curve.jpg', dpi=400)


