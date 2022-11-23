"""sub_model: mRNA Nerual Network with fully connected layers"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class MrnaNet(nn.Module):
	"""Extract representations from mRNA modality"""
	def __init__(self, mrna_length, m_length):
		super(MrnaNet, self).__init__()

		# Linear Layers
		self.mrna_hidden1 = nn.Linear(mrna_length, 1200)
		self.mrna_hidden2 = nn.Linear(1200, 800)
		self.mrna_hidden3 = nn.Linear(800, m_length)

		# Batch Normalization Layers
		self.bn1 = nn.BatchNorm1d(1200)
		self.bn2 = nn.BatchNorm1d(800)
		self.bn3 = nn.BatchNorm1d(m_length)

		#Dropout_layer
		self.dropout_layer1 = nn.Dropout()

	def forward(self, mrna_input):
		mrna = torch.tanh(self.bn1(self.mrna_hidden1(mrna_input)))
		mrna = torch.tanh(self.bn2(self.mrna_hidden2(mrna)))
		mrna = torch.tanh(self.bn3(self.mrna_hidden3(mrna)))

		mrna = self.dropout_layer1(mrna)

		return mrna