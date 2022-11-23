"""sub_model: miRNA Nerual Network with highway gate"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sub_models.highway import Highway


class MirnaNet(nn.Module):
	"""Extract representations from miRNA modality"""
	def __init__(self, mirna_length, m_length, device=None):
		"""miRNA Nerual Network with fully connected layers
		Parameters
		----------
		mirna_length: int
			The input dimension of miRNA modality.

		m_length: int
			Output dimension.

		"""
		super(MirnaNet, self).__init__()
		self.m_length = m_length
		self.device = device
		# Linear Layers
		self.mirna_hidden1 = nn.Linear(mirna_length, m_length)
		self.highway = Highway(m_length, 10, f=F.relu)

		# Batch Normalization Layers
		self.bn1 = nn.BatchNorm1d(m_length)

		# Dropout Layer
		self.dropout_layer1 = nn.Dropout(p=0.3)


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


	def forward(self, mirna_input):
		mask = self._make_mask(mirna_input.shape, 0.1)
		mirna = mirna_input.mul(mask)
		
		mirna = self.bn1(self.mirna_hidden1(mirna.float()))
		mirna = self.highway(mirna)
		mirna = torch.sigmoid(self.dropout_layer1(mirna))

		return mirna
























