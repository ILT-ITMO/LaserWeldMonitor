from torch import nn
from typing import Tuple
import torch.nn.functional as F
import torch


class NNClassifier(nn.Module):
    def __init__(self, config: Tuple[int, int] = (8, 64, 32, 1)) -> None:
        super().__init__()
        module_list = []
        for i in range(len(config) - 2):
            module_list.append(nn.Linear(config[i], config[i + 1]))
            module_list.append(nn.ReLU())
            module_list.append(nn.Dropout())
        module_list.append(nn.Linear(config[-2], config[-1]))  # last layer
        self.layers = nn.Sequential(*module_list)

    def forward(self, x):
        x = self.layers(x)
        return F.sigmoid(x)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        #out = F.relu(out)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return F.sigmoid(out)
    