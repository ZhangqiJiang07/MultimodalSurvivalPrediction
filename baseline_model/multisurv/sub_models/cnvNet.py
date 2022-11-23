"""sub_model: CNV Nerual Network with fully connected layers"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class CnvNet(nn.Module):
	"""Extract representations from CNV modality"""
	def __init__(self, cnv_length, m_length):
		"""CNV Nerual Network with fully connected layers
		Parameters
		----------
		cnv_length: int
			The input dimension of miRNA modality.

		m_length: int
			Output dimension.

		"""
		super(CnvNet, self).__init__()

		# Linear Layers
		self.cnv_hidden1 = nn.Linear(cnv_length, 2048)
		self.cnv_hidden2 = nn.Linear(2048, 1500)
		self.cnv_hidden3 = nn.Linear(1500, 1024)
		self.cnv_hidden4 = nn.Linear(1024, 800)
		self.cnv_hidden5 = nn.Linear(800, m_length)

		# Batch Normalization Layers
		self.bn1 = nn.BatchNorm1d(2048)
		self.bn2 = nn.BatchNorm1d(1500)
		self.bn3 = nn.BatchNorm1d(1024)
		self.bn4 = nn.BatchNorm1d(800)
		self.bn5 = nn.BatchNorm1d(m_length)

		# Dropout Layer
		self.dropout_layer1 = nn.Dropout()

	def forward(self, cnv_input):
		cnv = torch.tanh(self.bn1(self.cnv_hidden1(cnv_input)))
		cnv = torch.tanh(self.bn2(self.cnv_hidden2(cnv)))
		cnv = torch.tanh(self.bn3(self.cnv_hidden3(cnv)))
		cnv = torch.tanh(self.bn4(self.cnv_hidden4(cnv)))
		cnv = torch.tanh(self.bn5(self.cnv_hidden5(cnv)))

		cnv = self.dropout_layer1(cnv)

		return cnv
