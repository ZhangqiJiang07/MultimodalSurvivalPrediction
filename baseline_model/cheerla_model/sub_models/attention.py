"""sub_model: Attention-based multimodal fusion"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
	def __init__(self, m_length, modalities, device=None):
		"""Attention-based multimodal fusion Part
		Parameters
		----------
		m_length: int
			Weight vector length, corresponding to the modality representation
			length.

		modalities: list
			The list of used modality.

		"""
		super(Attention, self).__init__()
		self.m_length = m_length
		self.data_modalities = modalities
		self.device = device
		# contrast a pipeline for different modality weight matrix
		self.pipeline = {}
		for modality in self.data_modalities:
			self.pipeline[modality] = nn.Linear(self.m_length, self.m_length, bias=False).to(self.device)

	def _scale_for_missing_modalities(self, x, out):
		batch_dim = x.shape[1]
		for i in range(batch_dim):
			patient = x[:, i, :]
			zero_dims = 0
			for modality in patient:
				if modality.sum().data == 0:
					zero_dims += 1

			if zero_dims > 0:
				scaler = zero_dims + 1
				out[i, :] = scaler * out[i, :]

		return out

	def forward(self, multimodal_input):
		"""
		multimodal_input: dictionary
			A dictionary of used modality data, like:
			{'clinical':tensor(sample_size, m_length), 'mRNA':tensor(,),}
		"""
		attention_weight = tuple()
		multimodal_features = tuple()
		for modality in self.data_modalities:
			attention_weight += (torch.tanh(self.pipeline[modality](multimodal_input[modality])),)
			multimodal_features += (multimodal_input[modality],)

		# Across feature
		attention_matrix = F.softmax(torch.stack(attention_weight), dim=0)
		fused_vec = torch.sum(torch.stack(multimodal_features) * attention_matrix, dim=0)

		fused_vec = self._scale_for_missing_modalities(torch.stack(multimodal_features), fused_vec)
		return fused_vec





