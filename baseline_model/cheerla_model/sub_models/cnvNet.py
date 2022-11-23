"""sub_model: CNV Nerual Network with fully connected layers"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sub_models.highway import Highway

class CnvNet(nn.Module):
	"""Extract representations from CNV modality"""
	def __init__(self, cnv_length, m_length, device=None):
		"""CNV Nerual Network with fully connected layers
		Parameters
		----------
		cnv_length: int
			The input dimension of miRNA modality.

		m_length: int
			Output dimension.

		"""
		super(CnvNet, self).__init__()
		self.device = device

		# Linear Layers
		self.cnv_hidden1 = nn.Linear(cnv_length, m_length)
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

	def forward(self, cnv_input):
		mask = self._make_mask(cnv_input.shape, 0.1)
		cnv = cnv_input.mul(mask)

		cnv = self.bn1(self.cnv_hidden1(cnv.float()))
		cnv = self.highway(cnv)
		cnv = torch.sigmoid(self.dropout_layer1(cnv))

		return cnv






