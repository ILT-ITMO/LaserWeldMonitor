from utils import prepare_dataset, prepare_no_tracker_data_from_split, QUALITY
import matplotlib.pyplot as plt
from scipy.stats import kstest, normaltest, spearmanr
import numpy as np
import seaborn as sns
import pandas as pd


classes = ['hu', 'hg', 'he','hp', 'hs', 'hm', 'hi']
videos = list(QUALITY.keys())
val_videos = videos[:10]
train_videos = videos[10:]
train_df, val_df = prepare_no_tracker_data_from_split('thermograms_analysis/metrics/without_tracker.json', val_videos, train_videos,
                                                      window_size=16, overlap_ratio=0.5)
df = pd.concat((val_df, train_df), axis=0)

target = df[classes]
df = df.drop(classes, axis=1)
print(df.head())
# data, y = prepare_dataset('thermograms_analysis/metrics/metrics_40.json', clf=False)
# print(data)

### Normal distribution test ###
# fig, axes = plt.subplots(2, 4, figsize=(18, 10))
columns = df.columns

# for ax, col in zip(axes.flatten(), data.columns):
#     ax.scatter(data[col], y)
#     ax.set_title(col, style='italic')
#     ax.set_xlabel('Feature')
#     ax.set_ylabel('Defect hi')
#     col_data = data[col].to_numpy()
#     ksstat, p_value = kstest(col_data, "norm", args=(col_data.mean(), col_data.std()))
#     print(f"P value of column {col} is {p_value}")


# plt.savefig('thermograms_analysis/figures/data_distribution.jpg')
# #plt.show()
# plt.close()

### Correlation test ###
# data = np.concatenate((data.to_numpy(), y.to_numpy()[..., None]), axis=1)

# corr = spearmanr(data, axis=0)
# print(corr.statistic)

#classes = ['hi']
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

import pandas as pd

df = pd.DataFrame(out_corr)
df.to_excel('thermograms_analysis/corr.xlsx')

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
print(out_p)
plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('thermograms_analysis/figures/correlation.jpg')