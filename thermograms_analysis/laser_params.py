from utils import prepare_dataset_laser_params
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt


X, y = prepare_dataset_laser_params('hi')

print(X.shape, y.shape)
print(X[0], y[0])

model = LogisticRegression()
model.fit(X, y)


xx, yy = np.mgrid[X[:,0].min() - 10 : X[:,0].max() + 10 : 1, X[:,1].min() - 100 : X[:,1].max() + 100 : 10]
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.figure()
C = plt.contour(xx, yy, probs, levels=[.5], vmin=0, vmax=.6, label='Decision boundary')
C.collections[0].set_label('Decision boundary')
plt.scatter(X[y==1,0], X[y==1, 1], c='red', s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1, label='With defect')
plt.scatter(X[y==0,0], X[y==0, 1], c='blue', s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1, label='Without defect')
plt.xlabel('V, mm/sec')
plt.ylabel('P_L, W')
plt.legend()
plt.savefig('thermograms_analysis/figures/laser_parameters.jpg')
plt.show()