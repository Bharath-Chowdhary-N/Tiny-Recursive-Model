"""
Purpose: To create embedding for the given input
Input: [Batch_size, total_entries]
Output: [Batch_size, total_entries, emb_dim]

# Definition
total_entries: For sudoku of 30x30, 900 is the total entries
vocab_size: 10 assuming numbers are 0-9
emb_dim: 512 (defined in the paper), also should be sufficient for this problem.
"""
import torch
import torch.nn as nn
import numpy as np

class Embedding(nn.Module):
    def __init__(self, vocab_size:int=10, emb_dim:int=512)->torch.Tensor:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
    def forward(self,x):
        """
        Inp:
            x (torch.tensor): [Batch_size, inp_size] 
        Returns:
            [Batch_size, total_entries, emb_dim]
            
        """
        return self.embedding(x)
        
        
        
        