"""Contrastive Loss and Similarity Loss"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class Loss(nn.Module):
	"""docstring for Loss"""
	def __init__(self):
		super(Loss, self).__init__()


	def _negative_log_likelihood_loss(self, pred_hazard, event, time):
		risk = pred_hazard['hazard']
		_, idx = torch.sort(time, descending=True)
		event = event[idx]
		risk = risk[idx].squeeze()

		hazard_ratio = torch.exp(risk)
		log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
		uncensored_likelihood = risk - log_risk
		censored_likelihood = uncensored_likelihood * event

		num_observed_events = torch.sum(event)
		neg_likelihood = -torch.sum(censored_likelihood) / num_observed_events

		return neg_likelihood


	def forward(self, representation, modalities, pred_hazard, event, time):
		loss = self._negative_log_likelihood_loss(pred_hazard, event, time)

		return loss
















