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
modalities = ['clinical', 'mRNA']
data_path = utils.DATA_PATH

# choose learning rate
if modalities[0] == 'clinical' or modalities[0] == 'CNV':
	lr = 0.01
else:
	lr = 0.0005

# setup random seed
utils.setup_seed(SEED)
# detect cuda
device = utils.test_gpu()

# selected cancer types
tested_cancer_type = ['BLCA', 'BRCA', 'CESC', 'COADREAD', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG',
						'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'PRAD', 'SKCM', 'STAD', 'THCA', 'UCEC']

cancer_size = {'BLCA':411, 'BRCA':1096, 'CESC':307, 'COADREAD':628, 'HNSC':527,
				'KICH':112, 'KIRC':537, 'KIRP':290, 'LAML':186, 'LGG':514,
				'LIHC':376, 'LUAD':513, 'LUSC':498, 'OV':582, 'PAAD':185,
				'PRAD':500, 'SKCM':455, 'STAD':437, 'THCA':507, 'UCEC':547}

# Cancer type embedded index
cancer_type_dic = {'ACC': 0, 'BLCA': 1, 'BRCA': 2, 'CESC': 3, 'CHOL': 4, 'COAD': 5, 'DLBC': 6, 'ESCA': 7, 'GBM': 8, 'HNSC': 9,
					'KICH': 10, 'KIRC': 11, 'KIRP': 12, 'LAML': 13, 'LGG': 14, 'LIHC': 15, 'LUAD': 16, 'LUSC': 17, 'MESO': 18,
					'OV': 19, 'PAAD': 20, 'PCPG': 21, 'PRAD': 22, 'READ': 23, 'SARC': 24, 'SKCM': 25, 'STAD': 26, 'TGCT': 27,
					'THCA': 28, 'THYM': 29, 'UCEC': 30, 'UCS': 31, 'UVM': 32}


mydataset = MyDataset(modalities, data_path)
prepro_clin_data_X, _, prepro_clin_data_y, _ = preprocess_clinical_data(data_path['clinical'])
prepro_clin_data_X.reset_index(drop=True)
prepro_clin_data_y.reset_index(drop=True)
prepro_clin_data_y['indicator'] = pd.DataFrame(prepro_clin_data_X[[1]].values * 2 + prepro_clin_data_y[[6]].values, dtype=int)

result_dic = {}

for cancer_type in tested_cancer_type:
	result_dic[cancer_type] = {}

	if cancer_type == 'COADREAD':
		cancer_type_list = [cancer_type_dic['COAD'], cancer_type_dic['READ']]
	else:
		cancer_type_list = [cancer_type_dic[cancer_type]]

	cancer_sample_true_index = np.array(prepro_clin_data_X.loc[prepro_clin_data_X[1].isin(cancer_type_list), :].index)

	trainVal_test_strtfdKFold = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)

	trainVal_test_kfold = trainVal_test_strtfdKFold.split(prepro_clin_data_X.loc[prepro_clin_data_X[1].isin(cancer_type_list), :], prepro_clin_data_y.loc[prepro_clin_data_X[1].isin(cancer_type_list), :][[6]])

	test_c_index_arr = []
	for _, test in trainVal_test_kfold:
		test_sampler = cancer_sample_true_index[test]
		train_val_X, train_val_y = prepro_clin_data_X.drop(index=test_sampler), prepro_clin_data_y.drop(index=test_sampler)

		test_size = int(cancer_size[cancer_type] - len(test_sampler))

		_, train_val_sameN_X, _, train_val_sameN_y = train_test_split(train_val_X, train_val_y, test_size=test_size, random_state=SEED, stratify=train_val_y[['indicator']])
		train_X, val_X, _, _ = train_test_split(train_val_sameN_X, train_val_sameN_y, test_size=0.25, random_state=SEED, stratify=train_val_sameN_y[[6]])
		train_sampler, val_sampler = train_X.index.tolist(), val_X.index.tolist()

		dataloaders = utils.get_dataloaders(mydataset, train_sampler, val_sampler, test_sampler, BATCH_SIZE)
		# Create survival model
		survmodel = Model(
			modalities=modalities,
			m_length=m_length,
			dataloaders=dataloaders,
			fusion_method='attention',
			trade_off=0.3,
			mode='total', # only_cox
			device=device)
		# Generate run tag
		run_tag = utils.compose_run_tag(
			model=survmodel, lr=lr, dataloaders=dataloaders,
			log_dir='.training_logs/', suffix=''
		)

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
		for data, data_label in dataloaders['test']:
			out, event, time = survmodel.predict(data, data_label)
			hazard, representation = out
			test_c_index = concordance_index(time.cpu().numpy(), -hazard['hazard'].detach().cpu().numpy(), event.cpu().numpy())
			test_c_index_arr.append(test_c_index.item())
		print(f'C-index on Test set: ', test_c_index.item())

	m, s = utils.evaluate_model(test_c_index_arr)
	result_dic[cancer_type]['values'] = test_c_index_arr
	result_dic[cancer_type]['Mean'] = m
	result_dic[cancer_type]['Std'] = s
	print('Mean and std: ', (m, s))


with open('figure3_same_number_clinicalmRNA.txt', 'w') as file:
	for k in result_dic:
		file.write(f'{k}************\n')
		for p in result_dic[k]:
			file.write(f'{p}: {result_dic[k][p]}\n')

file.close()
























