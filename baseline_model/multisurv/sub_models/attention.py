"""Attention-based multimodal fusion"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
	def __init__(self, m_length, device=None):
		super(Attention, self).__init__()
		self.m_length = m_length
		self.device = device
		# contrast a pipeline for different modality weight matrix
		self.fusion_layer = nn.Linear(self.m_length, self.m_length, bias=False).to(self.device)

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
		ndata = tuple()
		for modality in multimodal_input:
			ndata += (multimodal_input[modality],)
		scores = self.fusion_layer(torch.stack(ndata))
		attention_weights = F.softmax(scores, dim=0)
		fused_vec = torch.sum(torch.stack(ndata)*attention_weights, dim=0)
		out = self._scale_for_missing_modalities(torch.stack(ndata), fused_vec)

		return out






