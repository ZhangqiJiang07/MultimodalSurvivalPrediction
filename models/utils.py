"""Utils"""

import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


# The dictionary of the input data path
DATA_PATH = {'clinical': '/mnt/preprocess/preprocessed_data/Pc_clinical_emb.csv',
			'mRNA': '/mnt/preprocess/preprocessed_data/PC_mRNA_threshold_7.csv',
			'miRNA': '/mnt/preprocess/preprocessed_data/PC_miRNA.csv',
			'CNV': '/mnt/preprocess/preprocessed_data/PC_CNV_threshold_20.csv'}


def setup_seed(seed):
	"""
	Set random seed for torch.
	"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True

def test_gpu():
	"""
	Detect the hardware: GPU or CPU?
	"""
	print('GPU？', torch.cuda.is_available())
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('The device is ', device)
	return device

def evaluate_model(c_index_arr):
	"""
	Calculate the Mean and the Standard Deviation about c-index array.
	"""
	m = np.sum(c_index_arr, axis=0) / len(c_index_arr)
	s = np.std(c_index_arr)
	return m, s



def get_dataloaders(mydataset, train_sampler, val_sampler, test_sampler, batch_size):
	"""
	Parameters
	----------
	mydataset: Dataset

	train_sampler: array
		Patient indexs in train set.

	val_sampler: array
		Patient indexs in validation set.

	test_sampler: array
		Patient indexs in test set.

	batch_size: int
		Number of patients in each batch.

	Return
	------
	A dictionary of train/validation/test set, like
		{'train': train_loader, 'val': val_loader, 'test': test_loader}
	"""
	
	dataloaders = {}
	dataloaders['train'] = DataLoader(mydataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_sampler))
	dataloaders['val'] = DataLoader(mydataset, batch_size=len(val_sampler), sampler=SubsetRandomSampler(val_sampler))
	dataloaders['test'] = DataLoader(mydataset, batch_size=len(test_sampler), sampler=SubsetRandomSampler(test_sampler))

	print('Dataset sizes (# patients):')
	print('train: ', len(train_sampler))
	print('  val: ', len(val_sampler))
	print(' test: ', len(test_sampler))
	print()
	print('Batch size: ', batch_size)

	return dataloaders


def compose_run_tag(model, lr, dataloaders, log_dir, suffix=''):
	"""
	Make the tag about modality and learning rate.
	"""
	def add_string(string, addition, sep='_'):
		if not string:
			return addition
		else: return string + sep + addition

	data = None
	for modality in model.data_modalities:
		data = add_string(data, modality)

	run_tag = f'{data}_lr{lr}'

	run_tag += suffix

	print(f'Run tag: "{run_tag}"')

	tb_log_dir = os.path.join(log_dir, run_tag)

	return run_tag


def save_5fold_results(c_index_arr, run_tag):
	"""
	Save the results after 5 fold cross validation.
	"""
	m, s = evaluate_model(c_index_arr)
	with open(f'proposed_{run_tag}.txt', 'w') as file:
		file.write(str(c_index_arr))
		file.write(f"\n Mean: {m}")
		file.write(f"\n Std: {s}")
	file.close()

























