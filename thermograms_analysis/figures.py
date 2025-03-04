import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from modules import min_loc_LoG

# Draw filtered frame
frames = np.load('thermograms_analysis/data/thermogram_7.npy')
t_min = frames.min()
t_max = frames.max()
frames -= t_min
frames = frames / (t_max - t_min)
frames *= 255
frames = frames.astype(np.uint8)

frame = frames[132]
filtered = min_loc_LoG(frame, 9)

f_min, f_max = filtered.min(), filtered.max()
filtered -= f_min
filtered /= (f_max - f_min)
filtered *= 255

binarized = ((filtered > 140) * 255).astype(np.uint8)
tracked = cv2.imread('thermograms_analysis/spatters_tracks.png', 0)

f, axs = plt.subplots(2,2, figsize=(8, 6))

images = (frame, filtered, binarized, tracked)
titles = ('a', 'b', 'c', 'd')
for ax, img, t in zip(axs.flatten(), images, titles):
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(t, style='italic')

#f.subplots_adjust(wspace=0., hspace=0.15, right=0.9, left=0.1, top=0.9, bottom=0.1)
plt.tight_layout()

plt.savefig('thermograms_analysis/spatters_tracking_algorithm.jpg')
plt.show()