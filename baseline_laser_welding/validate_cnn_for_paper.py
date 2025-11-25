import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, precision_recall_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary
from utils import make_datasets, make_conv2d_datasets
from models import resnet3d, Conv2DClassifier

fold_1 = ['thermogram_21.npy', 'thermogram_20.npy', 'thermogram_25.npy', 'thermogram_30.npy', 'thermogram_8.npy', 'thermogram_31.npy']
fold_2 = ['thermogram_22.npy', 'thermogram_11.npy', 'thermogram_10.npy', 'thermogram_16.npy', 'thermogram_26.npy', 'thermogram_12.npy']
fold_3 = ['thermogram_18.npy', 'thermogram_23.npy', 'thermogram_32.npy', 'thermogram_14.npy', 'thermogram_13.npy', 'thermogram_7.npy']
fold_4 = ['thermogram_33.npy', 'thermogram_17.npy', 'thermogram_9.npy', 'thermogram_34.npy', 'thermogram_19.npy', 'thermogram_24.npy']

folds = [['thermogram_21.npy', 'thermogram_20.npy', 'thermogram_25.npy', 'thermogram_30.npy', 'thermogram_8.npy', 'thermogram_31.npy'],
         ['thermogram_22.npy', 'thermogram_11.npy', 'thermogram_10.npy', 'thermogram_16.npy', 'thermogram_26.npy', 'thermogram_12.npy'],
         ['thermogram_18.npy', 'thermogram_23.npy', 'thermogram_32.npy', 'thermogram_14.npy', 'thermogram_13.npy', 'thermogram_7.npy'],
         ['thermogram_33.npy', 'thermogram_17.npy', 'thermogram_9.npy', 'thermogram_34.npy', 'thermogram_19.npy', 'thermogram_24.npy']]

model_weights = ['weights/fold_1_3d_ws32_best_f1_0.7586.pt', 'weights/fold_2_3d_ws32_best_f1_0.8545.pt', 
                 'weights/fold_3_3d_ws32_best_f1_0.8457.pt', 'weights/fold_4_3d_ws32_best_f1_0.8747.pt']
# model_weights = ['weights/fold_1_2d_best_f1_0.7094.pt', 'weights/fold_2_2d_best_f1_0.8177.pt',
#                  'weights/fold_3_2d_best_f1_0.8375.pt', 'weights/fold_4_2d_best_f1_0.8844.pt']

classes = ('hg', 'hs', 'hm', 'hi')
BS = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = resnet3d('resnet18')
# model = Conv2DClassifier(1, len(classes))
model.to(device)
model.eval()
f1_folds = []
prec_folds = []
rec_folds = []
y_real = []
y_proba = []
map_folds = []
ap_folds = []
for i in tqdm(range(4), total=4):
    aps = []
    _, val_ds = make_datasets('markup.json', 'dataset', classes, w_size=32, overlap=0.5, 
                                     val_videos=folds[i])
    # _, val_ds = make_conv2d_datasets('markup.json', 'dataset', classes, 
    #                                  val_videos=folds[i])
    val_dl = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=4)
    model.load_state_dict(torch.load(model_weights[i]))
    total_preds = []
    total_true = []
    with tqdm(val_dl, unit='batch') as tepoch:
        for i, (X_val, y_val) in enumerate(tepoch):
            tepoch.set_description(f"FOLD {i} VALIDATING")
            X_val, y_val = X_val.to(device), y_val.to(device)  # get data
            with torch.autocast('cuda', torch.float16), torch.no_grad():
                y_pred = model(X_val)  # get predictions

            total_preds.append(F.sigmoid(y_pred).cpu().numpy())
            total_true.append(y_val.cpu().numpy())

    total_preds = np.concatenate(total_preds)
    total_true = np.concatenate(total_true)

    mAP = 0
    for cl_id, cl in enumerate(classes):
        ap = average_precision_score(total_true[:, cl_id], total_preds[:, cl_id])
        aps.append(ap)
        mAP += ap
    map_folds.append(mAP / len(classes))
    ap_folds.append(aps)
    pr, rec, f1, _ = precision_recall_fscore_support(total_true, (total_preds > 0.55).astype(int), average=None)
    f1_folds.append(f1)
    prec_folds.append(pr)
    rec_folds.append(rec)
    y_real.append(total_true)
    y_proba.append(total_preds)


print(summary(model, X_val.shape[1:], device='cuda'))

# for i in range(4):
#     f1 = f1_folds[i]
#     for cl_id, cl in enumerate(classes):
#         print(f"FOLD {i}. F1-score for class {cl}: {f1[cl_id]:.3f}")

map_folds = np.array(map_folds)
print(f"Fold-wise mAP: {map_folds.mean():.3f} +- {map_folds.std():.3f}")

f1_folds = np.array(f1_folds)
prec_folds = np.array(prec_folds)
rec_folds = np.array(rec_folds)
ap_folds = np.array(ap_folds)
print(f"Precision: {np.round(prec_folds.mean(axis=0), 3)} +- {np.round(prec_folds.std(axis=0), 3)} {np.round(prec_folds.mean(), 3)}")
print(f"Recall: {np.round(rec_folds.mean(axis=0), 3)} +- {np.round(rec_folds.std(axis=0), 3)} {np.round(rec_folds.mean(), 3)}")
print(f"F1-score: {np.round(f1_folds.mean(axis=0), 3)} +- {np.round(f1_folds.std(axis=0), 3)} {np.round(f1_folds.mean(), 3)}")
print(f"AP: {np.round(ap_folds.mean(axis=0), 3)} +- {np.round(ap_folds.std(axis=0), 3)} {np.round(ap_folds.mean(), 3)}")
# y_real = np.concatenate(y_real)
# y_proba = np.concatenate(y_proba)
# pr, rec, f1, _ = precision_recall_fscore_support(y_real, (y_proba > 0.5).astype(int), average=None)
# print(f"Precision: {pr} macro {pr.mean()}")
# print(f"Recall: {rec} macro {rec.mean()}")
# print(f"F1-score: {f1} macro {f1.mean()}")



# for cl_id, cl in enumerate(classes):
#     print(f"ACROSS FOLDS. F1-score for class {cl}: {f1[cl_id]:.3f}")
# ap_sum = 0
# for cl_id, cl in enumerate(classes):
#     pr, rec, _ = precision_recall_curve(y_real[:, cl_id], y_proba[:, cl_id])
#     ap = average_precision_score(y_real[:, cl_id], y_proba[:, cl_id])
#     ap_sum += ap
#     label = f"{cl} AP={ap:.3f}"
#     plt.plot(rec, pr, label=label)

# pr, rec, _ = precision_recall_curve(y_real.flatten(), y_proba.flatten())
# label = f"Total mAP={round(ap_sum / len(classes), 3)}"
# plt.plot(rec, pr, linewidth=2, label=label, color='black')
# plt.xlim((0, 1))
# plt.ylim((0, 1))
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend()
# plt.savefig('PR_curve.png', dpi=300)