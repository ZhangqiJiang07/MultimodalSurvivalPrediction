"""sub_model: mRNA Nerual Network with fully connected layers"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class MrnaNet(nn.Module):
	"""Extract representations from mRNA modality"""
	def __init__(self, mrna_length, m_length):
		"""miRNA Nerual Network with fully connected layers
		Parameters
		----------
		mrna_length: int
			The input dimension of mRNA modality.

		m_length: int
			Output dimension.

		"""
		super(MrnaNet, self).__init__()

		# Linear Layers
		self.mrna_hidden1 = nn.Linear(mrna_length, m_length)
		# self.mrna_hidden2 = nn.Linear(800, 400)
		# self.mrna_hidden2 = nn.Linear(400, m_length)

		# Batch Normalization Layers
		# self.bn1 = nn.BatchNorm1d(800)
		# self.bn1 = nn.BatchNorm1d(400)
		self.bn1 = nn.BatchNorm1d(m_length)

		#Dropout_layer
		# self.dropout_layer1 = nn.Dropout()
		self.dropout_layer2 = nn.Dropout(p=0.4)

	def forward(self, mrna_input):
		mrna = torch.relu(self.bn1(self.mrna_hidden1(mrna_input)))
		# mrna = self.dropout_layer1(mrna)
		# mrna = torch.relu(self.bn2(self.mrna_hidden2(mrna)))
		# mrna = torch.relu(self.bn3(self.mrna_hidden3(mrna)))

		mrna = self.dropout_layer2(mrna)

		return mrna