from scipy.stats import spearmanr
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt


def compute_corr(feat: pd.DataFrame, gt: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    corr = np.zeros((len(gt.columns), len(feat.columns)))
    p_value = np.zeros((len(gt.columns), len(feat.columns)))

    for j, f in enumerate(list(feat.columns)):
        for i, t in enumerate(list(gt.columns)):
            sp = spearmanr(gt[t], feat[f])
            corr[i, j] = sp[0]
            p_value[i, j] = sp[1]

    return corr, p_value


def validate_model_plot_kfold(model, X: pd.DataFrame, y: pd.DataFrame, cls: str = 'hi') -> Dict:
    output = {}
    k_fold = StratifiedKFold(n_splits=10, shuffle=True)
    y_real = []
    y_proba = []
    y = y[cls]
    y = y.to_numpy().astype(bool)
    for i, (train_index, test_index) in enumerate(k_fold.split(X, y)):
        Xtrain, Xtest = X.to_numpy()[train_index], X.to_numpy()[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        model.fit(Xtrain, ytrain)
        pred_proba = model.predict_proba(Xtest)[:, 1]
        precision, recall, _ = precision_recall_curve(ytest, pred_proba)
        lab = 'Fold %d AP=%.3f' % (i+1, average_precision_score(ytest, pred_proba))
        precision = precision.tolist()
        precision.append(1)
        precision.insert(0, 0)
        recall = recall.tolist()
        recall.append(0)
        recall.insert(0, 1)

        output[lab] = [precision, recall]
        y_real.append(ytest)
        plt.plot(precision, recall, label=lab)
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
    plt.plot(precision, recall, label=lab, lw=2, color='black')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend(loc='lower left', fontsize='small')
    plt.savefig('photodiode_analysis/figures/pr_curve.png')
    return output


def validate_model_plot(model, X_train: pd.DataFrame, y_train: pd.DataFrame, 
                        X_val: pd.DataFrame, y_val: pd.DataFrame, cls: str = 'hi') -> Dict:
    output = {}
    y_train = y_train[cls]
    y_train = (y_train.to_numpy() > 0).astype(int)
    y_val = y_val[cls]
    y_val = (y_val.to_numpy() > 0).astype(int)
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, pred_proba)
    lab = 'AP=%.3f' % (average_precision_score(y_val, pred_proba))
    precision = precision.tolist()
    precision.append(1)
    precision.insert(0, 0)
    recall = recall.tolist()
    recall.append(0)
    recall.insert(0, 1)
    output[lab] = [precision, recall]
    plt.plot(precision, recall, label=lab)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend(loc='lower left', fontsize='small')
    plt.savefig('photodiode_analysis/figures/pr_curve_split.png')
    return output
