"""MyDataset"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler


def FindOutIndex(data):
	out_idx = []
	target = data.values.tolist()
	for i in range(data.shape[0]):
		try:
			target_ex = np.array(target[i], dtype='int')
		except:
			out_idx.append(i)
	return out_idx


def preprocess_clinical_data(clinical_path):
	data_clinical = pd.read_csv(clinical_path, header=None)
	target_data = data_clinical[[6, 7]]
	out_idx = FindOutIndex(target_data)

	clin_variables = data_clinical[[1, 2, 3, 4, 5]]
	idx = clin_variables[clin_variables[5].isnull()].index
	g = [i for i in idx]

	data_clinical.drop(index=out_idx+g, inplace=True)

	target_data = data_clinical[[6, 7]]
	clin_data_categorical = data_clinical[[1, 2, 3, 4]]
	clin_data_continuous = data_clinical[[5]]

	return clin_data_categorical, clin_data_continuous, target_data, out_idx+g


class MyDataset(torch.utils.data.Dataset):
	def __init__(self, modalities, data_path):
		"""
		Parameters
		----------
		modalities: list
			Used modalities

		data_path: dict
			The path of used data.

		Returns
		-------
		data: dictionary
			{'clin_data_categorical': ,..,'mRNA': ...}

		data_label: dictionary
			{'label':[[event, time]]}
		"""
		super(MyDataset, self).__init__()
		self.data_modalities = modalities
		# label
		clin_data_categorical, clin_data_continuous, target_data, remove_idx = preprocess_clinical_data(data_path['clinical'])
		self.target = target_data.values.tolist()

		# clinical
		if 'clinical' in self.data_modalities:
			self.clin_cat = clin_data_categorical.values.tolist()
			self.clin_cont = clin_data_continuous.values.tolist()

		# mRNA
		if 'mRNA' in self.data_modalities:
			data_mrna = pd.read_csv(data_path['mRNA'], header=None)
			data_mrna.drop(index=remove_idx, inplace=True)
			self.data_mrna = data_mrna.values.tolist()

		# miRNA
		if 'miRNA' in self.data_modalities:
			data_mirna = pd.read_csv(data_path['miRNA'], header=None)
			data_mirna.drop(index=remove_idx, inplace=True)
			self.data_mirna = data_mirna.values.tolist()

		# CNV
		if 'CNV' in self.data_modalities:
			data_cnv = pd.read_csv(data_path['CNV'], header=None)
			data_cnv.drop(index=remove_idx, inplace=True)
			self.data_cnv = data_cnv.values.tolist()

	def __len__(self):
		return len(self.clin_cat)

	def __getitem__(self, index):
		data = {}
		data_label = {}
		target_y = np.array(self.target[index], dtype='int')
		target_y = torch.from_numpy(target_y)
		data_label['label'] = target_y.type(torch.LongTensor)

		
		if 'clinical' in self.data_modalities:
			clin_cate = np.array(self.clin_cat[index]).astype(np.int64)
			clin_cate = torch.from_numpy(clin_cate)
			data['clinical_categorical'] = clin_cate

			clin_conti = np.array(self.clin_cont[index]).astype(np.float32)
			clin_conti = torch.from_numpy(clin_conti)
			data['clinical_continuous'] = clin_conti


		if 'mRNA' in self.data_modalities:
			mrna = np.array(self.data_mrna[index])
			mrna = torch.from_numpy(mrna)
			data['mRNA'] = mrna.type(torch.FloatTensor)


		if 'miRNA' in self.data_modalities:
			mirna = np.array(self.data_mirna[index])
			mirna = torch.from_numpy(mirna)
			data['miRNA'] = mirna.type(torch.FloatTensor)

		if 'CNV' in self.data_modalities:
			cnv = np.array(self.data_cnv[index])
			cnv = torch.from_numpy(cnv)
			data['CNV'] = cnv.type(torch.FloatTensor)

		return data, data_label

































