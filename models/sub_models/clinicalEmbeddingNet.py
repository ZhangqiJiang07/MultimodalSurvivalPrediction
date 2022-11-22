"""sub_model: Clinical Embedding Network"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClinicalEmbeddingNet(nn.Module):
	def __init__(self, m_length, n_continuous=1, embedding_size=[(33, 17), (2, 1), (6, 3), (145, 50)]):

		"""Clinical Embedding Network
		Parameters
		----------
		n_continuous: int (Default:1)
			The number of continuous variables.

		m_length: int
			Representation length.

		embedding_size: list (Default:[(33, 17), (2, 1), (6, 3), (145, 50)])
			Embedding size = (original categorical dimension, embedded dimension)

		"""
		super(ClinicalEmbeddingNet, self).__init__()
		# Embedding Layer
		self.embedding_layers = nn.ModuleList([nn.Embedding(categories, size)
												for categories, size in embedding_size])

		n_emb = sum(e.embedding_dim for e in self.embedding_layers)
		self.n_emb, self.n_continuous = n_emb, n_continuous

		# Linear Layer
		self.hidden1 = nn.Linear(self.n_emb + self.n_continuous, m_length)
		# batch normalization
		self.bn1 = nn.BatchNorm1d(self.n_continuous)
		self.emb_drop = nn.Dropout(0.4)

	def forward(self, x_categorical, x_continuous):
		x = [e(x_categorical[:, i]) for i, e in enumerate(self.embedding_layers)]
		x = torch.cat(x, 1)
		x = self.emb_drop(x)

		x2 = self.bn1(x_continuous)

		x = torch.cat([x, x2], 1)
		x = self.hidden1(x)

		return x







