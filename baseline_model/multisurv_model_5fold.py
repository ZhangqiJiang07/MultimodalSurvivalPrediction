
"""5 fold cross validation for Multisurv"""
import sys
sys.path.append('/mnt/multisurv')

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index
from data_loader import MyDataset
from data_loader import preprocess_clinical_data
import utils
from model import Model
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


m_length = 512
BATCH_SIZE = 60
EPOCH = 30
LR = 0.001
K = 5
SEED = 24

data_path = utils.DATA_PATH

modalities_list = [['clinical'], ['mRNA'], ['miRNA'], ['CNV'],
['clinical', 'CNV'], ['clinical', 'mRNA'], ['clinical', 'miRNA'], ['clinical', 'mRNA', 'CNV'],
['clinical', 'miRNA', 'CNV'], ['clinical', 'miRNA', 'mRNA'],
['clinical', 'miRNA', 'mRNA', 'CNV']]

# setup random seed
utils.setup_seed(SEED)
# detect cuda
device = utils.test_gpu()

for modalities in modalities_list:
	# create dataset
	mydataset = MyDataset(modalities, data_path)

	# create sampler
	prepro_clin_data_X, _, prepro_clin_data_y, _ = preprocess_clinical_data(data_path['clinical'])
	prepro_clin_data_X.reset_index(drop=True)
	prepro_clin_data_y.reset_index(drop=True)
	train_testVal_strtfdKFold = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
	train_testVal_kfold = train_testVal_strtfdKFold.split(prepro_clin_data_X, prepro_clin_data_y[[6]])

	test_c_index_arr = []
	for k, (train, test_val) in enumerate(train_testVal_kfold):
		# Create Train/validation/Test DataLoaders
		x_val, x_test, _, _ = train_test_split(prepro_clin_data_X.iloc[test_val, :], prepro_clin_data_y.iloc[test_val, :][[6]], test_size=0.5, random_state=24, stratify=prepro_clin_data_y.iloc[test_val, :][[6]])
		val, test = list(x_val.index), list(x_test.index)
		dataloaders = utils.get_dataloaders(mydataset, train, val, test, BATCH_SIZE)

		# Create survival model
		survmodel = Model(
			modalities=modalities,
			m_length=m_length,
			dataloaders=dataloaders,
			device=device)
		# Generate run tag
		run_tag = utils.compose_run_tag(
			model=survmodel, lr=LR, dataloaders=dataloaders,
			log_dir='.training_logs/', suffix=''
		)

		fit_args = {
		'num_epochs': EPOCH,
		'lr': LR,
		'info_freq': 2,
		'log_dir': os.path.join('.training_logs/', run_tag),
		'lr_factor': 0.5,
		'scheduler_patience': 10,
		}
		# model fitting
		survmodel.fit(**fit_args)

		# Load the best weights on validation set and test the model performance on test set!
		survmodel.test()
		for data, data_label in dataloaders['test']:
			out, event, time = survmodel.predict(data, data_label)
			hazard, representation = out
			test_c_index = concordance_index(time.cpu().numpy(), -hazard['hazard'].detach().cpu().numpy(), event.cpu().numpy())
			test_c_index_arr.append(test_c_index.item())
		print(f'C-index on Test set: ', test_c_index.item())

	print('Mean and std: ', utils.evaluate_model(test_c_index_arr))
	utils.save_5fold_results(test_c_index_arr, 'multisurv', run_tag)









