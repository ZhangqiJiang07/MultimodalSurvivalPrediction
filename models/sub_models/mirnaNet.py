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
		self.mirna_hidden1 = nn.Linear(mirna_length, 400)
		self.mirna_hidden2 = nn.Linear(400, m_length)

		# Batch Normalization Layers
		self.bn1 = nn.BatchNorm1d(400)
		self.bn2 = nn.BatchNorm1d(m_length)

		# Dropout Layer
		self.dropout_layer1 = nn.Dropout(p=0.3)
		self.dropout_layer2 = nn.Dropout(p=0.4)

	def forward(self, mirna_input):
		mirna = torch.relu(self.bn1(self.mirna_hidden1(mirna_input)))
		mirna = self.dropout_layer1(mirna)
		mirna = torch.relu(self.bn2(self.mirna_hidden2(mirna)))

		mirna = self.dropout_layer2(mirna)

		return mirna
