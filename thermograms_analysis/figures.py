import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from utils import find_optimal_threshold, prepare_dataset, validate_model_plot
from catboost import CatBoostClassifier

from modules import min_loc_LoG

# Draw filtered frame
# frames = np.load('thermograms_analysis/data/thermogram_7.npy')
# t_min = frames.min()
# t_max = frames.max()
# frames -= t_min
# frames = frames / (t_max - t_min)
# frames *= 255
# frames = frames.astype(np.uint8)

# frame = frames[132]
# filtered = min_loc_LoG(frame, 9)

# f_min, f_max = filtered.min(), filtered.max()
# filtered -= f_min
# filtered /= (f_max - f_min)
# filtered *= 255

# binarized = ((filtered > 140) * 255).astype(np.uint8)

# cv2.imwrite('thermograms_analysis/figures/frame.jpg', frame)
# cv2.imwrite('thermograms_analysis/figures/filtered.jpg', filtered)
# cv2.imwrite('thermograms_analysis/figures/binarized.jpg', binarized)

# tracked = cv2.imread('thermograms_analysis/spatters_tracks.png', 0)

# f, axs = plt.subplots(2,2, figsize=(8, 6))

# images = (frame, filtered, binarized, tracked)
# titles = ('a', 'b', 'c', 'd')
# for ax, img, t in zip(axs.flatten(), images, titles):
#     ax.imshow(img, cmap='gray')
#     ax.axis('off')
#     ax.set_title(t, style='italic')

# #f.subplots_adjust(wspace=0., hspace=0.15, right=0.9, left=0.1, top=0.9, bottom=0.1)
# plt.tight_layout()

# plt.savefig('thermograms_analysis/figures/spatters_tracking_algorithm.jpg')
# plt.show()



model = CatBoostClassifier(logging_level='Silent')
#model = NNClassifier()

X, y = prepare_dataset('thermograms_analysis/metrics/metrics_40.json')
out = validate_model_plot(model, X, y)

model = CatBoostClassifier(logging_level='Silent')
th = find_optimal_threshold(model, 'thermograms_analysis/metrics/metrics_40.json')
print(th)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

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

for k, v in out.items():
    pr, rec = v

    if k.startswith('Fold'):
        axes[0].plot(pr, rec, label=k, lw=1)
    else:
        axes[0].plot(pr, rec, label='Overall AP=0.778', lw=3, color='black')

axes[0].set_xlabel('Precision')
axes[0].set_ylabel('Recall')
axes[0].legend()
axes[0].grid()

axes[0].set_title('a')
axes[0].axis('equal')
axes[0].set(xlim=(0, 1), ylim=(0, 1))

plt.savefig('thermograms_analysis/figures/Figure_6.png', dpi=400)
plt.show()
