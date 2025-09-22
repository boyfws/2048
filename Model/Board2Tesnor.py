import torch.nn as nn
import numpy as np 
import torch

class Board2Tensor(nn.Module):
    def __init__(self, emb_dim: int = 10):
        """
        У нас есть возможные значения -
        0, 2, 4, 8, 16, 32, 64, 128,
        256, 512, 1024, 2048
        """
        super().__init__()
        self.emb = nn.Embedding(
            16, emb_dim
        )


    def forward(self, X: np.ndarray):
        """
        X - набор полей c shape (batch_size, 4, 4)
        """
        X = X.reshape(X.shape[0], 16, order='C')

        new_X = np.ones_like(X, dtype=np.int32)
        mask = X != 0
        new_X[mask] = X[mask] 
        
        X = np.log2(new_X).astype(int)
        X = torch.from_numpy(X).long().to(self.emb._parameters["weight"].device)
        X_emb = self.emb(X) # (batch_size, 16, emb_dim)
        concatenated_emb = X_emb.view(X_emb.size(0), -1)  # (batch_size, 16 * emb_dim)
        
        return concatenated_emb




