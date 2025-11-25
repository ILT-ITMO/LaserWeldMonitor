from utils import prepare_sequences_dataset, seed_everything, QUALITY, train_conv1d_mlb_clf
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_recall_fscore_support
import json
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from models import CNNTimeSeriesClassifier, HybridCNNTransformerMultilabelV2, MultiLayerBiLSTMClassifier, TransformerClassifier

# mAP 0.866 CNN-Transformer, BS 64, AdamW lr 0.0001, WS 64, default model
# mAP 0.834 CNN-default, BS 64, AdamW lr 0.0001, WS 64
# mAP 0.801 LSTM, BS 64, AdamW lr 0.0001, WS 64
# mAP 0.780 Transformer-default, BS 64, AdamW lr 0.0001, WS 64


from sklearn.preprocessing import StandardScaler

def scale_sequences(train_sequences, val_sequences):
    _, len_seq, n_feat = train_sequences.shape
    scaler = StandardScaler()
    scaler.fit(train_sequences.reshape(-1, n_feat))
    
    scaled_train = scaler.transform(train_sequences.reshape(-1, n_feat)).reshape(-1, len_seq, n_feat)
    scaled_val = scaler.transform(val_sequences.reshape(-1, n_feat)).reshape(-1, len_seq, n_feat)
    
    return scaled_train, scaled_val

seed_everything(42)

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
f1_folds = []
y_real = []
y_proba = []

for i, (train_index, val_index) in tqdm(enumerate(k_fold.split(list(labels.keys()), list(labels.values()))), total=folds):
    train_videos = [list(labels.keys())[v] for v in train_index]
    val_videos = [list(labels.keys())[v] for v in val_index]
    X_train, y_train, X_val, y_val = prepare_sequences_dataset('without_tracker.json', val_videos, train_videos, 
                                                               classes, sequence_length=64, overlap_ratio=0.5)
    y_val = (y_val > 0).astype(int)
    y_train = (y_train > 0).astype(int)
    # model = CNNTimeSeriesClassifier(X_train.shape[2], X_train.shape[1], len(classes), conv_filters=[64, 128, 256], kernel_sizes=[7, 5, 3],
    #              dropout_rate=0.3, use_batch_norm=True)
    model = HybridCNNTransformerMultilabelV2(X_train.shape[2], X_train.shape[1], len(classes))
    # model = MultiLayerBiLSTMClassifier(X_train.shape[2], 128, 4, len(classes))
    # model = TransformerClassifier(X_train.shape[2], X_train.shape[1], len(classes), d_model=256, nhead=8, num_layers=3, 
    #              dim_feedforward=512, dropout=0.1)
    name = f"fold_{i}_best_f1.pt"
    best_f1, y_pred = train_conv1d_mlb_clf(model, X_train, y_train, X_val, y_val, name)
    # print(f"Best F1-score for fold {i}: {best_f1:.3f}")
    pr, rec, f1, _ = precision_recall_fscore_support(y_val, (y_pred > 0.5).astype(int), average=None)
    f1_folds.append(f1)
    y_real.append(y_val)
    y_proba.append(y_pred)


for i in range(folds):
    f1 = f1_folds[i]
    for cl_id, cl in enumerate(classes):
        print(f"FOLD {i}. F1-score for class {cl}: {f1[cl_id]:.3f}")

y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
pr, rec, f1, _ = precision_recall_fscore_support(y_real, (y_proba > 0.5).astype(int), average=None)
for cl_id, cl in enumerate(classes):
    print(f"ACROSS FOLDS. F1-score for class {cl}: {f1[cl_id]:.3f}")
print(f"Macro F1 - {f1.mean():.3f}")
ap_sum = 0
for cl_id, cl in enumerate(classes):
    pr, rec, _ = precision_recall_curve(y_real[:, cl_id], y_proba[:, cl_id])
    ap = average_precision_score(y_real[:, cl_id], y_proba[:, cl_id])
    ap_sum += ap
    label = f"{cl} AP={ap:.3f}"
    plt.plot(rec, pr, label=label)

# pr, rec, _ = precision_recall_curve(y_real.flatten(), y_proba.flatten())
label = f"Total mAP={round(ap_sum / len(classes), 3)}"
print(label)
# plt.plot(rec, pr, linewidth=2, label=label, color='black')
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig('PR_curve.jpg', dpi=300)


