"""Abstract model"""


import os

from net import Net
import torch
from torch.optim import Adam
from loss import Loss
from model_coach import ModelCoach

class _BaseModelWithDataLoader:
	def __init__(self, modalities, m_length, dataloaders, fusion_method='attention', device=None):
		self.data_modalities = modalities
		self.m_length = m_length
		self.dataloaders = dataloaders
		self.device = device
		self.fusion_method = fusion_method

		self._instantiate_model()
		self.model_blocks = [name for name, _ in self.model.named_children()]



	def _instantiate_model(self, move2device=True):
		print('Instantiate Survival model...')
		self.model = Net(self.data_modalities, self.m_length, self.fusion_method, self.device)

		if move2device:
			self.model = self.model.to(self.device)


class Model(_BaseModelWithDataLoader):
	def __init__(self, modalities, m_length, dataloaders, fusion_method='attention', trade_off=0.3, mode='total', device=None):
		super().__init__(modalities, m_length, dataloaders, fusion_method, device)

		self.optimizer = Adam
		self.loss = Loss(trade_off=trade_off, mode=mode)

	def fit(self, num_epochs, lr, info_freq, log_dir, lr_factor=0.1, scheduler_patience=5):
		self._instantiate_model()
		optimizer = self.optimizer(self.model.parameters(), lr=lr)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer=optimizer, mode='max', factor=lr_factor,
			patience = scheduler_patience, verbose=True, threshold=1e-3,
			threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)


		model_coach = ModelCoach(
			model=self.model, modalities=self.data_modalities,
			dataloaders=self.dataloaders, optimizer=optimizer,
			criterion=self.loss, device=self.device)

		model_coach.train(num_epochs, scheduler, info_freq, log_dir)

		self.model = model_coach.model
		self.best_model_weights = model_coach.best_wts
		self.best_c_index = model_coach.best_perf
		self.current_c_index = model_coach.current_perf

	def save_weights(self, saved_epoch, prefix, weight_dir):
		print('Saving model weights to file:')
		if saved_epoch == 'current':
			epoch = list(self.current_concord.keys())[0]
			value = self.current_concord[epoch]
			file_name = os.path.join(
				weight_dir,
				f'{prefix}_{epoch}_c_index{value:.2f}.pth')
		else:
			file_name = os.path.join(
				weight_dir,
				f'{prefix}_{saved_epoch}_' + \
				f'c_index{self.best_concord_values[saved_epoch]:.2f}.pth')
			self.model.load_state_dict(self.best_model_weights[saved_epoch])

		torch.save(self.model.stat_dict(), file_name)
		print(' ', file_name)

	def test(self):
		self.model.load_state_dict(self.best_model_weights['best_wts'])
		self.model = self.model.to(self.device)

	def load_weights(self, path):
		print('Load model weights:')
		print(path)
		self.model.load_state_dict(torch.load(path))
		self.model = self.model.to(self.device)

	def predict(self, data, data_label):
		for modality in data:
			data[modality] = data[modality].to(self.device)
		event = data_label['label'][:, 0].to(self.device)
		time = data_label['label'][:, 1].to(self.device)

		return self.model(data), event, time





























