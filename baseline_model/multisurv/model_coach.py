""" The 'coach' guide the model to fit data"""


import os
import copy
import pandas as pd
import torch
from lifelines.utils import concordance_index
from torch.utils.tensorboard import SummaryWriter


class ModelCoach:
	def __init__(self, model, modalities, dataloaders, optimizer, criterion, device=None):
		self.model = model
		self.modalities = modalities
		self.dataloaders = dataloaders
		self.optimizer = optimizer
		self.criterion = criterion.to(device)

		# self.best_perf = {'epoch a': 0.0, 'epoch b': 0.0, 'epoch c': 0.0}
		# self.best_wts = {'epoch a': None, 'epoch b': None, 'epoch c': None}
		
		self.best_perf = {'best_score': 0.0}
		self.best_wts = {'best_wts': None}

		self.current_perf = {'epoch a': 0}
		self.device = device


	def _data2device(self, data):
		for modality in data:
			data[modality] = data[modality].to(self.device)

		return data

	def _compute_loss(self, representation, modalities, pred_hazard, event, time):
		loss = self.criterion(representation=representation, modalities=modalities, pred_hazard=pred_hazard, event=event, time=time)

		return loss

	def _log_info(self, phase, logger, epoch, epoch_loss, epoch_c_index):
		info = {phase + '_loss': epoch_loss,
				phase + '_c_index': epoch_c_index}

		for tag, value in info.items():
			logger.add_scalar(tag, value, epoch)

	def _process_data_batch(self, data, data_label, phase):
		"""
		Train model using a batch.
		"""
		data = self._data2device(data)
		event = data_label['label'][:, 0].to(self.device)
		time = data_label['label'][:, 1].to(self.device)

		with torch.set_grad_enabled(phase == 'train'):
			hazard, representation = self.model(data)
			loss = self._compute_loss(representation, self.modalities, hazard, event, time)

			if phase == 'train':
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

		return loss, hazard['hazard'], time, event


	def _run_training_loop(self, num_epochs, scheduler, info_freq, log_dir):
		logger = SummaryWriter(log_dir)
		log_info = True

		if info_freq is not None:
			def print_header():
				sub_header = ' Epoch     Loss     Ctd     Loss     Ctd'
				print('-' * (len(sub_header) + 2))
				print('             Training        Validation')
				print('           ------------     ------------')
				print(sub_header)
				print('-' * (len(sub_header) + 2))

			print()
			print_header()

		for epoch in range(1, num_epochs+1):
			if info_freq is None:
				print_info = False
			else:
				print_info = (epoch == 1) or (epoch % info_freq == 0)

			for phase in ['train', 'val']:
				if phase == 'train':
					self.model.train()
				else:
					self.model.eval()

				running_loss = []

				if print_info or log_info:
					running_sample_time = torch.FloatTensor().to(self.device)
					running_sample_event = torch.LongTensor().to(self.device)
					running_hazard = torch.FloatTensor().to(self.device)

				for data, data_label in self.dataloaders[phase]:
					loss, hazard, time, event = self._process_data_batch(data, data_label, phase)

					running_loss.append(loss.item())
					running_sample_time = torch.cat((running_sample_time, time.data.float()))
					running_sample_event = torch.cat((running_sample_event, event.long().data))
					running_hazard = torch.cat((running_hazard, hazard.detach()))

				epoch_loss = torch.mean(torch.tensor(running_loss))

				epoch_c_index = concordance_index(running_sample_time.cpu().numpy(), -running_hazard.cpu().numpy(), running_sample_event.cpu().numpy())

				if print_info:
					if phase == 'train':
						message = f' {epoch}/{num_epochs}'
					space = 10 if phase == 'train' else 27
					message += ' ' * (space - len(message))
					message += f'{epoch_loss:.4f}' 
					space = 19 if phase == 'train' else 36
					message += ' ' * (space - len(message))
					message += f'{epoch_c_index:.3f}' 

					if phase == 'val':
						print(message)


				if log_info:
					self._log_info(phase=phase, logger=logger, epoch=epoch,
									epoch_loss=epoch_loss, epoch_c_index=epoch_c_index)

				if phase == 'val':
					if scheduler:
						scheduler.step(epoch_c_index)

					# Record current performance
					k = list(self.current_perf.keys())[0]
					self.current_perf['epoch' + str(epoch)] = self.current_perf.pop(k)
					self.current_perf['epoch' + str(epoch)] = epoch_c_index

					# Record top best model
					# for k, v in self.best_perf.items():
					if epoch_c_index > self.best_perf['best_score']:
							# self.best_perf['best_score'] = self.best_perf.pop(k)
						self.best_perf['best_score'] = epoch_c_index
							# self.best_wts['best_wts'] = self.best_wts.pop(k)
						self.best_wts['best_wts'] = copy.deepcopy(self.model.state_dict())
							# break

	def train(self, num_epochs, scheduler, info_freq, log_dir):
		self._run_training_loop(num_epochs, scheduler, info_freq, log_dir)
		print('>>>>> Best validation C-indices:')
		for k, v in self.best_perf.items():
			print(f'     {v} ({k})')
































