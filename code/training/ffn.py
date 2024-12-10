import torch
import torch.nn as nn

class GestureNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes) -> None:
        super(GestureNet, self).__init__()
        self.network_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),         # 63 -> 128
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),                         # bath norm for better convergence
            nn.Dropout(0.3),                                                # prevent overfitting
            nn.Linear(hidden_size, hidden_size),        # 128 -> 128                          
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)         # 128 -> 7
        )
    
    def forward(self, x):
        logits = self.network_stack(x)
        return logits
        