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
		self.cnv_hidden1 = nn.Linear(cnv_length, 1300)
		self.cnv_hidden2 = nn.Linear(1300, 700)
		self.cnv_hidden3 = nn.Linear(700, 300)
		self.cnv_hidden4 = nn.Linear(300, m_length)

		# Batch Normalization Layers
		self.bn1 = nn.BatchNorm1d(1300)
		self.bn2 = nn.BatchNorm1d(700)
		self.bn3 = nn.BatchNorm1d(300)
		self.bn4 = nn.BatchNorm1d(m_length)

		# Dropout Layer
		self.dropout_layer1 = nn.Dropout(p=0.3)
		self.dropout_layer2 = nn.Dropout(p=0.3)
		self.dropout_layer3 = nn.Dropout(p=0.4)


	def forward(self, cnv_input):
		cnv = torch.relu(self.bn1(self.cnv_hidden1(cnv_input)))
		cnv = self.dropout_layer1(cnv)
		cnv = torch.relu(self.bn2(self.cnv_hidden2(cnv)))
		cnv = torch.relu(self.bn3(self.cnv_hidden3(cnv)))
		cnv = self.dropout_layer2(cnv)
		cnv = torch.relu(self.bn4(self.cnv_hidden4(cnv)))

		cnv = self.dropout_layer3(cnv)

		return cnv
