import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import torch.nn.functional as F
from typing import Tuple
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import os
import time
from torch.cuda.amp import GradScaler

from utils import make_datasets, make_conv2d_datasets
from models import resnet3d, Conv2DClassifier

fold_1 = ['thermogram_21.npy', 'thermogram_20.npy', 'thermogram_25.npy', 'thermogram_30.npy', 'thermogram_8.npy', 'thermogram_31.npy']
fold_2 = ['thermogram_22.npy', 'thermogram_11.npy', 'thermogram_10.npy', 'thermogram_16.npy', 'thermogram_26.npy', 'thermogram_12.npy']
fold_3 = ['thermogram_18.npy', 'thermogram_23.npy', 'thermogram_32.npy', 'thermogram_14.npy', 'thermogram_13.npy', 'thermogram_7.npy']
fold_4 = ['thermogram_33.npy', 'thermogram_17.npy', 'thermogram_9.npy', 'thermogram_34.npy', 'thermogram_19.npy', 'thermogram_24.npy']

if __name__ == '__main__':
    classes = ('hg', 'hs', 'hm', 'hi')
    EPOCHS = 15
    BS = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = resnet3d('resnet18')
    model = Conv2DClassifier(1, len(classes))
    model.to(device)
    # train_ds, val_ds = make_datasets('markup.json', 'dataset', classes, w_size=32, overlap=0.5, 
    #                                  val_videos=fold_4)
    train_ds, val_ds = make_conv2d_datasets('markup.json', 'dataset', classes, 
                                     val_videos=fold_4)
    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=4)
    train_loss = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.00007)
    scaler = torch.amp.GradScaler()
    best_f1 = 0

    for epoch in range(EPOCHS):
        model.train()  # set model to training mode
        total_loss = 0  # init total losses and metrics
        with tqdm(train_dl, unit='batch') as tepoch:
            for i, (X_train, y_train) in enumerate(tepoch):
                tepoch.set_description(f"EPOCH {epoch + 1}/{EPOCHS} TRAINING")
                X_train, y_train = X_train.to(device), y_train.to(device)  # get data
                # print(X_train.size())
                # print(y_train.size())
                with torch.autocast('cuda', torch.float16):
                    y_pred = model(X_train)  # get predictions
                    loss = train_loss(y_pred, y_train)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                total_loss += loss.item()
                tepoch.set_postfix(loss=total_loss / (i + 1))
                time.sleep(0.1)
        
        total_loss = total_loss / (i + 1)
        model.eval()
        total_val_loss = 0
        total_preds = []
        total_true = []
        with tqdm(val_dl, unit='batch') as tepoch:
            for i, (X_val, y_val) in enumerate(tepoch):
                tepoch.set_description(f"EPOCH {epoch + 1}/{EPOCHS} VALIDATING")
                X_val, y_val = X_val.to(device), y_val.to(device)  # get data
                with torch.autocast('cuda', torch.float16), torch.no_grad():
                    y_pred = model(X_val)  # get predictions
                    loss = train_loss(y_pred, y_val)

                total_preds.append(F.sigmoid(y_pred).cpu().numpy())
                total_true.append(y_val.cpu().numpy())
                total_val_loss += loss.item()

        total_val_loss /= (i + 1)

        print(f"TRAIN LOSS: {total_loss:.4f}")
        print(f"VAL LOSS: {total_val_loss:.4f}")
        total_preds = (np.concatenate(total_preds) > 0.5).astype(int)
        total_true = np.concatenate(total_true)
        pr, rec, f1, _ = precision_recall_fscore_support(total_true, total_preds, average=None)
        for cl_id, cl in enumerate(classes):
            print(f"F1-score for class {cl}: {f1[cl_id]:.3f}")
        
        pr, rec, f1, _ = precision_recall_fscore_support(total_true, total_preds, average='macro')

        if best_f1 < f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_f1.pt')

        
        
        print("Best F1: ", best_f1)


