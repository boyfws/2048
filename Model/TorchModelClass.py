import torch.nn as nn
import numpy as np 
import torch
from .Board2Tesnor import Board2Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlockLinear(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.0):
        """
        Residual блок с Linear слоями для одномерных данных
        
        Args:
            in_features: количество входных признаков
            out_features: количество выходных признаков
            dropout_rate: вероятность dropout
        """
        super().__init__()
        
        # Основная ветка
        self.linear1 = nn.Linear(in_features, out_features)
        self.norm1 = nn.BatchNorm1d(out_features)
        
        self.linear2 = nn.Linear(out_features, out_features)
        self.norm2 = nn.BatchNorm1d(out_features)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Shortcut connection
    
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        

        self.activation = nn.LeakyReLU(0.1)

    
    def forward(self, x):
        # Сохраняем исходный тензор для shortcut
        identity = x
        
        # Основная ветка
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.norm2(out)
        out = self.dropout(out)
        
        # Shortcut connection
        shortcut = self.shortcut(identity)
        
        # Сложение и финальная активация
        out += shortcut
        out = self.activation(out)
        
        return out



class TorchModel(nn.Module):

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.emb_dim = emb_dim

        self.board_conv = Board2Tensor(emb_dim=self.emb_dim)

        self.model = nn.Sequential(
            ResidualBlockLinear(16 * self.emb_dim, 16 * self.emb_dim),
            ResidualBlockLinear(16 * self.emb_dim, 256, 0.05),
            ResidualBlockLinear(256, 256, 0.1),
            ResidualBlockLinear(256, 256, 0.2),
            ResidualBlockLinear(256, 128, 0.1),
            ResidualBlockLinear(128, 64, 0.1),
            nn.Linear(64, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, X: np.ndarray):
        embs = self.board_conv(X)
        return self.model(embs) 

    


