import json
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from utils import prepare_dataset_from_video_split, QUALITY


with open('thermograms_analysis/metrics/metrics_40.json', 'r') as f:
    videos = list(json.load(f).keys())

labels = {}
for vid in videos:
    label = 0
    for d in QUALITY[vid]:
        if d[-1] > 1:
            label = 1

    labels[vid] = label

print(labels)
k_fold = StratifiedKFold(2)
output = {}
y_real = []
y_proba = []
for i, (train_index, val_index) in tqdm(enumerate(k_fold.split(list(labels.keys()), list(labels.values()))), total=5):
    train_videos = [list(labels.keys())[v] for v in train_index]
    val_videos = [list(labels.keys())[v] for v in val_index]
    train_df, y_train, val_df, y_val = prepare_dataset_from_video_split('thermograms_analysis/metrics/metrics_40.json', train_videos, val_videos)
    print(val_videos)
    model = CatBoostClassifier(logging_level='Silent')
    # model = RandomForestClassifier()
    # model = make_pipeline(PolynomialFeatures(2), StandardScaler(), LogisticRegression())
    # model = make_pipeline(StandardScaler(), SVC(probability=True))
    model.fit(train_df, y_train)
    pred_proba = model.predict_proba(val_df)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, pred_proba)
    lab = 'Fold %d AP=%.3f' % (i+1, average_precision_score(y_val, pred_proba))
    precision = precision.tolist()
    precision.append(1)
    precision.insert(0, 0)
    recall = recall.tolist()
    recall.append(0)
    recall.insert(0, 1)

    output[lab] = [precision, recall]
    y_real.append(y_val)
    plt.plot(recall, precision, label=lab)
    y_proba.append(pred_proba)

y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
precision, recall, _ = precision_recall_curve(y_real, y_proba)
lab = 'Overall AP=%.3f' % (average_precision_score(y_real, y_proba))
precision = precision.tolist()
precision.append(1)
precision.insert(0, 0)
recall = recall.tolist()
recall.append(0)
recall.insert(0, 1)
output[lab] = [precision, recall]
plt.plot(recall, precision, label=lab, lw=2, color='black')
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.legend(loc='lower left', fontsize='small')
plt.show()

# val_videos = ['thermogram_16.npy', 'thermogram_20.npy', 'thermogram_14.npy', 'thermogram_23.npy', 'thermogram_25.npy', 'thermogram_11.npy', 'thermogram_31.npy', 'thermogram_17.npy']
# train_videos = [vid for vid in videos if vid not in val_videos]
# train_df, y_train, val_df, y_val = prepare_dataset_from_video_split('thermograms_analysis/metrics/metrics_35.json', train_videos, val_videos)
# model = CatBoostClassifier(logging_level='Silent')
# model.fit(train_df, y_train)
# pred_proba = model.predict_proba(val_df)[:, 1]
# precision, recall, _ = precision_recall_curve(y_val, pred_proba)
# precision = precision.tolist()
# precision.append(1)
# precision.insert(0, 0)
# recall = recall.tolist()
# recall.append(0)
# recall.insert(0, 1)
# plt.plot(recall, precision, lw=2, color='black')
# plt.ylabel('Precision')
# plt.xlabel('Recall')
# plt.show()