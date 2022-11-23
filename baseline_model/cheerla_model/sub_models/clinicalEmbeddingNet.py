"""sub_model: Clinical Embedding Network"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClinicalEmbeddingNet(nn.Module):
	def __init__(self, m_length, n_continuous=1, embedding_size=[(33, 17), (2, 1), (6, 3), (145, 50)], device=None):

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
		self.device = device

		# Linear Layer
		self.hidden1 = nn.Linear(self.n_emb + self.n_continuous, m_length)
		# batch normalization
		self.bn1 = nn.BatchNorm1d(self.n_continuous)
		self.emb_drop = nn.Dropout(0.4)

	def _make_mask(self, sample_size, prob):
		mask_one_zero = lambda x: 1 if x > prob else 0
		# 0-1 Distribution
		vec = np.random.rand(sample_size[0], 1)
		mask_vec = np.array(list(map(mask_one_zero, vec)))
		mask_vec = mask_vec.reshape(sample_size[0], 1)
		# 构造mask
		one_vec = np.ones((1, sample_size[1]))
		mask = mask_vec * one_vec
		mask = torch.from_numpy(mask).to(self.device)

		return mask

	def forward(self, x_categorical, x_continuous):
		x = [e(x_categorical[:, i]) for i, e in enumerate(self.embedding_layers)]
		x = torch.cat(x, 1)
		x = self.emb_drop(x)

		x2 = self.bn1(x_continuous)

		x = torch.cat([x, x2], 1)

		mask = self._make_mask(x.shape, 0.1)
		x = x.mul(mask)

		x = self.hidden1(x.float())

		return x







