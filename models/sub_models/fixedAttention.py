"""sub_model: Fixed Attention-based multimodal fusion"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedAttention(nn.Module):
	def __init__(self, m_length, modalities, device=None):
		"""Fixed Attention-based multimodal fusion Part
		Parameters
		----------
		m_length: int
			Weight vector length, corresponding to the modality representation
			length.

		modalities: list
			The list of used modality.

		"""
		super(FixedAttention, self).__init__()
		self.m_length = m_length
		self.data_modalities = modalities
		self.device = device
		# contrast a pipeline for different modality weight matrix

	def forward(self, multimodal_input):
		"""
		multimodal_input: dictionary
			A dictionary of used modality data, like:
			{'clinical':tensor(sample_size, m_length), 'mRNA':tensor(,),}
		"""
		attention_weight = tuple()
		multimodal_features = tuple()
		for modality in self.data_modalities:
			attention_weight += (torch.ones(multimodal_input[modality].shape[0], self.m_length).to(self.device),)
			multimodal_features += (multimodal_input[modality],)

		# Across feature
		attention_matrix = F.softmax(torch.stack(attention_weight), dim=0)
		fused_vec = torch.sum(torch.stack(multimodal_features) * attention_matrix, dim=0)

		return fused_vec





