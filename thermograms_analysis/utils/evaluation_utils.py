import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Literal, Union
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import nn
import torch
from copy import deepcopy
from .data_utils import prepare_dataset


def validate_model_plot(model, X: pd.DataFrame, y: pd.Series) -> Dict:
    output = {}
    k_fold = StratifiedKFold(n_splits=10, shuffle=True)
    y_real = []
    y_proba = []
    for i, (train_index, test_index) in enumerate(k_fold.split(X, y)):
        Xtrain, Xtest = X.to_numpy()[train_index], X.to_numpy()[test_index]
        ytrain, ytest = y.to_numpy()[train_index], y.to_numpy()[test_index]
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
    plt.show()
    return output


def validate_model(model, X: pd.DataFrame, y: pd.Series) -> float:
    k_fold = StratifiedKFold(n_splits=10, shuffle=True)
    y_real = []
    y_proba = []
    for i, (train_index, test_index) in enumerate(k_fold.split(X, y)):
        Xtrain, Xtest = X.to_numpy()[train_index], X.to_numpy()[test_index]
        ytrain, ytest = y.to_numpy()[train_index], y.to_numpy()[test_index]
        model.fit(Xtrain, ytrain)
        pred_proba = model.predict_proba(Xtest)[:, 1]
        y_real.append(ytest)
        y_proba.append(pred_proba)

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    return average_precision_score(y_real, y_proba)


def validate_list_models(models: Dict, data: List[str]) -> pd.DataFrame:
    out = {key: [] for key in data}
    columns = list(models.keys())
    for js in tqdm(data, total=len(data)):
        X, y = prepare_dataset(js, type='reduced')
        for _, model in tqdm(models.items(), total=len(models)):
            if isinstance(model, nn.Module):
                out[js].append(validate_nn_model(model, X, y))
            else:
                out[js].append(validate_model(model, X, y))

    return pd.DataFrame.from_dict(out, orient='index', columns=columns)


def find_optimal_threshold(model_, data: str) -> pd.DataFrame:
    X, y = prepare_dataset(data, type='reduced')
    k_fold = StratifiedKFold(n_splits=10, shuffle=True)
    y_real = []
    y_proba = []
    for i, (train_index, test_index) in enumerate(k_fold.split(X, y)):
        if isinstance(model_, nn.Module):
            X = StandardScaler().fit_transform(X)
            Xtrain, Xtest = X[train_index], X[test_index]
        else:
            Xtrain, Xtest = X.to_numpy()[train_index], X.to_numpy()[test_index]
        ytrain, ytest = y.to_numpy()[train_index], y.to_numpy()[test_index]
        
        if isinstance(model_, nn.Module):
            model = train_nn_clf(deepcopy(model_), Xtrain, ytrain)
            model.eval()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                pred_proba = model(torch.from_numpy(Xtest).to(device).float()).squeeze().cpu().numpy()
        else:
            model_.fit(Xtrain, ytrain)
            pred_proba = model_.predict_proba(Xtest)[:, 1]
        y_real.append(ytest)
        y_proba.append(pred_proba)

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    
    
    thresholds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    out = {'thresholds': thresholds, 'precision': [], 'recall': [], 'f1-score': []}
    for th in thresholds:
        pr, rec, f1, _ = precision_recall_fscore_support(y_real, (y_proba > th) * 1, average='binary')
        out['precision'].append(pr)
        out['recall'].append(rec)
        out['f1-score'].append(f1)

    out['thresholds'].append(1)
    out['precision'].append(1)
    out['recall'].append(0)
    out['f1-score'].append(0)
    return pd.DataFrame.from_dict(out)


### FOR PYTORCH FULLY CONNECTED NETWORKS


def train_nn_clf(model: nn.Module, X: np.ndarray, y: np.ndarray) -> nn.Module:
    ds = torch.cat((torch.from_numpy(X), torch.from_numpy(y)[..., None]), dim=1)  # creat dataset
    dl = DataLoader(ds, batch_size=256, shuffle=True)  # create dataloader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optim = Adam(model.parameters())
    train_loss = nn.BCELoss()
    model.train()
    for _ in range(300):
        for batch in dl:
            X_train, y_train = batch[:, :-1].to(device).float(), batch[:, -1].to(device).float()
            y_pred = model(X_train)
            loss = train_loss(y_pred, y_train[..., None])
            optim.zero_grad()
            loss.backward()  # back propogation
            optim.step()  # optimizer's step
    return model


def validate_nn_model_plot(model_: nn.Module, X: pd.DataFrame, y: pd.Series) -> None:
    k_fold = StratifiedKFold(n_splits=10, shuffle=True)
    y_real = []
    y_proba = []
    if not hasattr(model_, 'lstm'):
        X = StandardScaler().fit_transform(X)
        split = k_fold.split(X, y)
    else:
        for i in range(8):
            X[:, :, i] = (X[:, :, i] - X[:, :, i].mean()) / X[:, :, i].std()
        split = k_fold.split(X, y[:, 0])

    for i, (train_index, test_index) in enumerate(split):
        Xtrain, Xtest = X[train_index], X[test_index]
        if not hasattr(model_, 'lstm'):
            ytrain, ytest = y.to_numpy()[train_index], y.to_numpy()[test_index]
        else:
            ytrain, ytest = y[train_index], y[test_index]
            ytest = ytest[:, 0]
        if not hasattr(model_, 'lstm'):
            model = train_nn_clf(deepcopy(model_), Xtrain, ytrain)
        else:
            model = train_lstm_clf(deepcopy(model_), Xtrain, ytrain)
        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            pred_proba = model(torch.from_numpy(Xtest).to(device).float()).squeeze().cpu().numpy()
        
        precision, recall, _ = precision_recall_curve(ytest, pred_proba)
        lab = 'Fold %d AP=%.4f' % (i+1, average_precision_score(ytest, pred_proba))
        y_real.append(ytest)
        plt.plot(precision, recall, label=lab)
        y_proba.append(pred_proba)

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    lab = 'Overall AP=%.4f' % (average_precision_score(y_real, y_proba))
    plt.plot(precision, recall, label=lab, lw=2, color='black')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend(loc='lower left', fontsize='small')
    plt.show()


def validate_nn_model(model_: nn.Module, X: pd.DataFrame, y: pd.Series) -> float:
    k_fold = StratifiedKFold(n_splits=10, shuffle=True)
    y_real = []
    y_proba = []
    if not hasattr(model_, 'lstm'):
        X = StandardScaler().fit_transform(X)
        split = k_fold.split(X, y)
    else:
        for i in range(8):
            X[:, :, i] = (X[:, :, i] - X[:, :, i].mean()) / X[:, :, i].std()
        split = k_fold.split(X, y[:, 0])
    for i, (train_index, test_index) in enumerate(k_fold.split(X, y)):
        Xtrain, Xtest = X[train_index], X[test_index]
        if not hasattr(model_, 'lstm'):
            ytrain, ytest = y.to_numpy()[train_index], y.to_numpy()[test_index]
        else:
            ytrain, ytest = y[train_index], y[test_index]
        if not hasattr(model_, 'lstm'):
            model = train_nn_clf(deepcopy(model_), Xtrain, ytrain)
        else:
            model = train_lstm_clf(deepcopy(model_), Xtrain, ytrain)
        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            pred_proba = model(torch.from_numpy(Xtest).to(device).float()).squeeze().cpu().numpy()
        y_real.append(ytest)
        y_proba.append(pred_proba)

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    return average_precision_score(y_real, y_proba)


### FOR PYTORCH LSTM NETWORKS


def train_lstm_clf(model: nn.Module, X: np.ndarray, y: np.ndarray) -> nn.Module:
    ds = torch.cat((torch.from_numpy(X), torch.from_numpy(y)[..., None]), dim=-1)  # creat dataset
    dl = DataLoader(ds, batch_size=256, shuffle=True)  # create dataloader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optim = Adam(model.parameters())
    train_loss = nn.BCELoss()
    model.train()
    for _ in range(100):
        for batch in dl:
            X_train, y_train = batch[..., :-1].to(device).float(), batch[:, 0, -1].to(device).float()
            y_pred = model(X_train)
            loss = train_loss(y_pred, y_train[..., None])
            optim.zero_grad()
            loss.backward()  # back propogation
            optim.step()  # optimizer's step
    return model
