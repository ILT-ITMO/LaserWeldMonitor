from utils import prepare_dataset
import matplotlib.pyplot as plt
from scipy.stats import kstest, normaltest, spearmanr
import numpy as np
import seaborn as sns


data, y = prepare_dataset('thermograms_analysis/metrics_40.json', clf=False)
print(data)

### Normal distribution test ###
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
columns = data.columns

for ax, col in zip(axes.flatten(), data.columns):
    ax.scatter(data[col], y)
    ax.set_title(col, style='italic')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Defect hi')
    col_data = data[col].to_numpy()
    ksstat, p_value = kstest(col_data, "norm", args=(col_data.mean(), col_data.std()))
    print(f"P value of column {col} is {p_value}")


plt.savefig('thermograms_analysis/figures/data_distribution.jpg')
#plt.show()
plt.close()

### Correlation test ###
# data = np.concatenate((data.to_numpy(), y.to_numpy()[..., None]), axis=1)

# corr = spearmanr(data, axis=0)
# print(corr.statistic)
classes = ['hu', 'hg', 'he','hp', 'hs', 'hm', 'hi']
#classes = ['hi']
out_corr = []
for cl in classes:
    data, y = prepare_dataset('thermograms_analysis/metrics_40.json', class_id=cl, clf=False)
    data = data.to_numpy()
    y = y.to_numpy()
    for i in range(data.shape[1]):
        corr = spearmanr(data[:, i], y).statistic
        out_corr.append(corr)


out_corr = np.array(out_corr).reshape(len(classes), -1)

plt.figure(figsize=(10, 10))
ax = sns.heatmap(out_corr, linewidths=1, xticklabels=columns, yticklabels=classes, annot=True)
ax.set_xlabel('Features')
ax.set_ylabel('Defects')
plt.yticks(rotation=0)
plt.xticks(rotation=20)
plt.savefig('thermograms_analysis/figures/correlation.jpg')