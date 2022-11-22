"""Contrastive Loss and Similarity Loss"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import accumulate
from random import shuffle



class Loss(nn.Module):
	"""docstring for Loss"""
	def __init__(self, trade_off=0.3, mode='total'):
		"""
		Parameters
		----------
		trade_off: float (Default:0.3)
			To balance the unsupervised loss and cox loss.

		mode: str (Default:'total')
			To determine which loss is used.
		"""
		super(Loss, self).__init__()
		self.trade_off = trade_off
		self.mode = mode


	def _negative_log_likelihood_loss(self, pred_hazard, event, time):
		risk = pred_hazard['hazard']
		_, idx = torch.sort(time, descending=True)
		event = event[idx]
		risk = risk[idx].squeeze()

		hazard_ratio = torch.exp(risk)
		log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + 1e-6)
		uncensored_likelihood = risk - log_risk
		censored_likelihood = uncensored_likelihood * event

		num_observed_events = torch.sum(event) + 1e-6
		neg_likelihood = -torch.sum(censored_likelihood) / num_observed_events

		return neg_likelihood


	def _random_match(self, batch_size):
		idx = list(range(batch_size))
		split_size = int(batch_size * 0.5)
		shuffle(idx)
		x1, x2 = idx[:split_size], idx[split_size:]
		if len(x1) != len(x2):
			x1.append(x2[0])

		return x1, x2


	def _contrastive_loss1(self, x1_idx, x2_idx, representation, modalities, margin=0.2):
		"""
		Only one modality
		"""
		con_loss = 0
		modality = modalities[0]
		for idx1, idx2 in zip(x1_idx, x2_idx):
			dis_x_y = torch.cosine_similarity(representation[modality][idx1],
													representation[modality][idx2], dim=0)
			con_loss += torch.pow(torch.clamp(margin+dis_x_y, min=0.0), 2)

		return con_loss / len(x1_idx)


	def _contrastive_loss2(self, x1_idx, x2_idx, representation, modalities, margin=0.2):
		"""
		More than one modality
		"""
		con_loss = 0
		for idx1, idx2 in zip(x1_idx, x2_idx):
			dis_x_x = 0
			dis_y_y = 0
			for i in range(len(modalities)-1):
				for j in range(i+1, len(modalities)):
					dis_x_x += torch.cosine_similarity(representation[modalities[i]][idx1],
														representation[modalities[j]][idx1], dim=0)
					dis_y_y += torch.cosine_similarity(representation[modalities[i]][idx2],
														representation[modalities[j]][idx2], dim=0)
			dis_x_y = 0
			for modality in modalities:
				dis_x_y += torch.cosine_similarity(representation[modality][idx1],
													representation[modality][idx2], dim=0)
			con_loss += torch.pow(torch.clamp(margin+dis_x_y-0.5*dis_x_x-0.5*dis_y_y, min=0.0), 2)

		return con_loss / len(x1_idx)


	def _unsupervised_similarity_loss(self, representation, modalities, t=1):
		k = 0
		similarity_loss = 0
		if len(modalities) > 1:
			while k < t:
				x1_idx, x2_idx = self._random_match(representation[modalities[0]].shape[0])
				similarity_loss += self._contrastive_loss2(x1_idx, x2_idx, representation, modalities)
				k += 1
		else:
			while k < t:
				x1_idx, x2_idx = self._random_match(representation[modalities[0]].shape[0])
				similarity_loss += self._contrastive_loss1(x1_idx, x2_idx, representation, modalities)
				k += 1

		return similarity_loss / t


	def _cross_entropy_loss(self, pred_hazard, event):
		return F.nll_loss(pred_hazard['score'], event)


	def forward(self, representation, modalities, pred_hazard, event, time):
		"""
		When mode = 'total' we use the proposed loss function,
		mode = 'only_cox' we remove the unsupervised loss.
		"""
		if self.mode == 'total':
			loss = self._cross_entropy_loss(pred_hazard, event) + self._negative_log_likelihood_loss(pred_hazard, event, time) + self.trade_off * self._unsupervised_similarity_loss(representation, modalities)
		elif self.mode == 'only_cox':
			loss = self._negative_log_likelihood_loss(pred_hazard, event, time)

		return loss
















