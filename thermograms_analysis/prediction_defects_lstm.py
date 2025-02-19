from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch
from nn_classifiers import LSTMClassifier


from utils import *


df, is_defect = prepare_dataset('metrics_40_lstm.json', type='lstm')


print(df.shape)
print(is_defect.shape)

model = LSTMClassifier(8, 64)

validate_nn_model_plot(model, df, is_defect)
