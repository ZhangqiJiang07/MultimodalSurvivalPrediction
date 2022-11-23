"""sub_model: Average multimodal fusion"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AverageFusion(nn.Module):
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
		super(AverageFusion, self).__init__()
		self.m_length = m_length
		self.data_modalities = modalities
		self.device = device

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
		multimodal_features = tuple()
		for modality in self.data_modalities:
			multimodal_features += (multimodal_input[modality],)

		# Across feature
		fused_vec = torch.sum(torch.stack(multimodal_features), dim=0)

		fused_vec = self._scale_for_missing_modalities(torch.stack(multimodal_features), fused_vec)
		return fused_vec





