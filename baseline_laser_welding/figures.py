import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.stats import kstest, normaltest, spearmanr
import seaborn as sns
import pandas as pd
from utils import prepare_spatters_dataset


with open('without_tracker.json', 'r') as f:
    data = json.load(f)


### FOR PIPELINE FIGURE ###
# data = data['thermogram_21.npy']

# plt.figure(figsize=(10, 3))
# plt.plot(data['welding_zone_temp'][50:800], lw=2, color='black')
# plt.savefig('figures/features.jpg', dpi=300)


### CORRELATION HEATMAP ###
classes = ['hu', 'hg', 'he','hp', 'hs', 'hm', 'hi']
videos = list(data.keys())
val_videos = videos[:10]
train_videos = videos[10:]
X_train, y_train, X_val, y_val = prepare_spatters_dataset('without_tracker.json', val_videos, train_videos, 
                                                                  classes, window_size=64, overlap_ratio=0.5)

df = pd.concat((X_train, X_val), axis=0)
target = pd.concat((y_train, y_val), axis=0)
print(df.head())
columns = df.columns
out_corr = []
out_p = []
for cl in classes:
    # data, y = prepare_dataset('thermograms_analysis/metrics/metrics_40.json', class_id=cl, clf=False)
    y = target[cl]
    data = df.to_numpy()
    y = y.to_numpy()
    for i in range(data.shape[1]):
        corr = spearmanr(data[:, i], y)
        out_corr.append(corr.statistic)
        out_p.append(corr.pvalue)


out_corr = np.array(out_corr).reshape(len(classes), -1)
df = pd.DataFrame(out_corr)
df.to_excel('figures/corr.xlsx')

out_p = np.array(out_p).reshape(len(classes), -1)
# add anotations to heatmap
annot = out_corr.round(3).astype(str)
for p in (0.05, 0.01, 0.001):
    annot = np.where(out_p < p, np.char.add(annot, '*'), annot)
columns = ['\n'.join(col.split('_')) for col in columns]
plt.figure(figsize=(20, 10))
ax = sns.heatmap(out_corr, linewidths=1, xticklabels=columns, yticklabels=classes, annot=annot, fmt='', square=True, cbar_kws={"shrink": 0.65})
ax.set_xlabel('Features')
ax.set_ylabel('Defects')
# print(out_p)
plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('figures/correlation.jpg', dpi=400)


out_corr = spearmanr(df, axis=0).statistic
plt.figure(figsize=(20, 10))
ax = sns.heatmap(out_corr, linewidths=1, xticklabels=columns, yticklabels=columns, square=True, cbar_kws={"shrink": 0.65})
ax.set_xlabel('Features')
ax.set_ylabel('Features')
plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('figures/auto_correlation.jpg', dpi=400)