"""sub_model: miRNA Nerual Network with fully connected layers"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class MirnaNet(nn.Module):
	"""Extract representations from miRNA modality"""
	def __init__(self, mirna_length, m_length):
		"""miRNA Nerual Network with fully connected layers
		Parameters
		----------
		mirna_length: int
			The input dimension of miRNA modality.

		m_length: int
			Output dimension.

		"""
		super(MirnaNet, self).__init__()

		# Linear Layers
		self.mirna_hidden1 = nn.Linear(mirna_length, 600)
		self.mirna_hidden2 = nn.Linear(600, m_length)

		# Batch Normalization Layers
		self.bn1 = nn.BatchNorm1d(600)
		self.bn2 = nn.BatchNorm1d(m_length)

		# Dropout Layer
		self.dropout_layer1 = nn.Dropout()

	def forward(self, mirna_input):
		mirna = torch.tanh(self.bn1(self.mirna_hidden1(mirna_input)))
		mirna = torch.tanh(self.bn2(self.mirna_hidden2(mirna)))

		mirna = self.dropout_layer1(mirna)

		return mirna
