# %pip install lifelines

import sys
sys.path.append('/mnt/models')

import numpy as np
import pandas as pd
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

# const
m_length = 128
BATCH_SIZE = 250
EPOCH = 30
lr = 0.01
K = 5
SEED = 24


data_path = utils.DATA_PATH

# selected modalities
modalities_list = [['clinical'], ['clinical', 'mRNA'], ['clinical', 'mRNA', 'CNV']]

# selected cancer types
tested_cancer_type = ['BLCA', 'BRCA', 'CESC', 'COADREAD', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG',
						'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'PRAD', 'SKCM', 'STAD', 'THCA', 'UCEC']

# Cancer type embedded index
cancer_type_dic = {'ACC': 0, 'BLCA': 1, 'BRCA': 2, 'CESC': 3, 'CHOL': 4, 'COAD': 5, 'DLBC': 6, 'ESCA': 7, 'GBM': 8, 'HNSC': 9,
					'KICH': 10, 'KIRC': 11, 'KIRP': 12, 'LAML': 13, 'LGG': 14, 'LIHC': 15, 'LUAD': 16, 'LUSC': 17, 'MESO': 18,
					'OV': 19, 'PAAD': 20, 'PCPG': 21, 'PRAD': 22, 'READ': 23, 'SARC': 24, 'SKCM': 25, 'STAD': 26, 'TGCT': 27,
					'THCA': 28, 'THYM': 29, 'UCEC': 30, 'UCS': 31, 'UVM': 32}

# setup random seed
utils.setup_seed(SEED)
# detect cuda
device = utils.test_gpu()


prepro_clin_data_X, _, prepro_clin_data_y, _ = preprocess_clinical_data(data_path['clinical'])
prepro_clin_data_X.reset_index(drop=True)
prepro_clin_data_y.reset_index(drop=True)
prepro_clin_data_y['indicator'] = pd.DataFrame(prepro_clin_data_X[[1]].values * 2 + prepro_clin_data_y[[6]].values, dtype=int)


for modalities in modalities_list:
	# choose learning rate
	if modalities[0] == 'clinical' or modalities[0] == 'CNV':
		lr = 0.01
	else:
		lr = 0.0005

	mydataset = MyDataset(modalities, data_path)
	# modality_dic = {}

	trainVal_test_stratifyKFold = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
	# cancer type stratify
	trainVal_test_kfold = trainVal_test_stratifyKFold.split(prepro_clin_data_X.drop(columns=[1]), prepro_clin_data_y['indicator'])

	# result container
	contain = {}
	for h in tested_cancer_type:
		contain[h] = []


	for train_val_cross, test_cross in trainVal_test_kfold:
		train_df, val_df, _, _ = train_test_split(prepro_clin_data_X.drop(columns=[1]).loc[train_val_cross, :], prepro_clin_data_y[[6]].loc[train_val_cross, :], test_size=0.25, random_state=SEED, stratify=prepro_clin_data_y[[6]].loc[train_val_cross, :])
		train_sampler, val_sampler = train_df.index.tolist(), val_df.index.tolist()
		dataloaders = utils.get_dataloaders(mydataset, train_sampler, val_sampler, test_cross, BATCH_SIZE)


		# Create survival model
		survmodel = Model(
			modalities=modalities,
			m_length=m_length,
			dataloaders=dataloaders,
			fusion_method='attention', #attention
			trade_off=0.3,
			mode='total', # only_cox
			device=device)

		run_tag = utils.compose_run_tag(
			model=survmodel, lr=lr, dataloaders=dataloaders,
			log_dir='.training_logs/', suffix='')

		fit_args = {
			'num_epochs': EPOCH,
			'lr': lr,
			'info_freq': 2,
			'log_dir': os.path.join('.training_logs/', run_tag),
			'lr_factor': 0.5,
			'scheduler_patience': 7,
			}

		# model fitting
		survmodel.fit(**fit_args)

		# Load the best weights on validation set and test the model performance on test set!
		survmodel.test()

		for cancer_type in tested_cancer_type:
		# map cancer type to crrospond embedded index
			if cancer_type == 'COADREAD':
				cancer_type_list = [cancer_type_dic['COAD'], cancer_type_dic['READ']]
			else:
				cancer_type_list = [cancer_type_dic[cancer_type]]

			cancer_test_sampler = prepro_clin_data_X.loc[test_cross, :].loc[prepro_clin_data_X.loc[test_cross, :][1].isin(cancer_type_list), :].index.tolist()
			dataloaders_cancer = utils.get_dataloaders(mydataset, [1], [1], cancer_test_sampler, BATCH_SIZE)
			for data, data_label in dataloaders_cancer['test']:
				out, event, time = survmodel.predict(data, data_label)
				hazard, representation = out
				test_c_index = concordance_index(time.cpu().numpy(), -hazard['hazard'].detach().cpu().numpy(), event.cpu().numpy())
				contain[cancer_type].append(test_c_index.item())

	# make dir
	dir_name = ''
	for j in modalities:
		dir_name += j

	with open(f'table4_proposed_{dir_name}.txt', 'w') as file:
		for k in contain:
			m, s = utils.evaluate_model(contain[k])
			file.write(f'{k}****************\n')
			file.write(f'Value: {contain[k]}\n')
			file.write(f'Mean: {m}\n')
			file.write(f'Std: {s}\n')
			file.write('\n')
	file.close()






































