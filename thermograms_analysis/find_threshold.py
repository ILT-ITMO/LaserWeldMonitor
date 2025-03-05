from utils import find_optimal_threshold
from catboost import CatBoostClassifier
from modules import NNClassifier
import matplotlib.pyplot as plt

model = CatBoostClassifier(logging_level='Silent')
#model = NNClassifier()

th = find_optimal_threshold(model, 'thermograms_analysis/metrics_40.json')
print(th)
plt.figure(figsize=(6, 6))

plt.plot(th['thresholds'], th['precision'], label='Precision', lw=2)
plt.plot(th['thresholds'], th['recall'], label='Recall', lw=2)
plt.plot(th['thresholds'], th['f1-score'], label='F1-score', lw=2)
plt.xlabel('Threshold')
plt.ylabel('Metric')
plt.legend()
plt.savefig('thermograms_analysis/figures/optimal_threshold_calculation.jpg')
plt.show()