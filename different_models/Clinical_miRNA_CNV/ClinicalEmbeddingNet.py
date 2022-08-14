import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalEmbedding(nn.Module):
    def __init__(self, embedding_sizes, n_cont, m_length):
        super().__init__()
        self.m_length = m_length
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings)
        self.n_emb, self.n_cont = n_emb, n_cont
        self.hidden1 = nn.Linear(self.n_emb + self.n_cont, self.m_length)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.emb_drop = nn.Dropout(0.4)

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn1(x_cont)
        x = torch.cat([x, x2], 1)
        x = F.relu(self.hidden1(x))
        return x