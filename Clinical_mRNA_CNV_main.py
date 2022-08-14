import sys
sys.path.append('different_models/Clinical_mRNA_CNV')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index
from itertools import *
from Clinical_mRNA_CNV_network import *
from ThreeModal_utils import *
from utils import *

#For Clinical, mRNA and CNV
PATH_CLIN = 'data/PC_Clinical.csv'
PATH_MRNA = 'data/PC_mRNA_0_1.csv'
PATH_CNV = 'data/PC_CNV_0_1.csv'
BATCH_SIZE = 3000
EPOCH = 30
K = 5



def main():
	setup_seed(7)
	device = test_gpu()
	embedding_size = [(33, 17), (2, 1), (6, 3), (145, 50)]
	print('Start training...')
	mydataset = MyDataset(PATH_MRNA, PATH_CNV, PATH_CLIN)
	folds = split_to_KFold(mydataset.__len__(), K)
	vali_c_index_arr = np.zeros([K, EPOCH])
	for j in range(K):
		'''5-Fold index'''
		print(f'**********{j+1}Fold*********')
		a = folds.copy()
		validation_sampler = SubsetRandomSampler(a[j])
		a.pop(j)
		train_sampler = SubsetRandomSampler(np.hstack(a))

		train_loader = DataLoader(mydataset, batch_size=BATCH_SIZE, sampler=train_sampler)
		validation_loader = DataLoader(mydataset, batch_size=len(validation_sampler), sampler=validation_sampler)
		
		#Create New Net
		net = AtteNet(mrna_lenth=1176, cnv_lenth=1739, embedding_sizes=embedding_size, n_cont=1, n_model=3, m_lenth=128)
		net.to(device)
		optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8)
		
		for epoch in range(EPOCH):
			i = 0
			mean_loss = 0
			target_time_data = 0
			target_event_data = 0
			pred_hazard_data = 0

			#Training set
			net.train()
			for train_step, (mrna_batch, cnv_batch, clin_cat_batch, clin_cont_batch, target_batch) in enumerate(train_loader):
				out, v = net(mrna=mrna_batch.to(device), cnv=cnv_batch.to(device), clin_cat=clin_cat_batch.to(device), clin_cont=clin_cont_batch.to(device))

				'''After the loop is completed, we start optimizing the parameters.'''
				loss = net.loss(out, target_batch.to(device), v, 1)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				'''Record the results and actual values of each batch to
				 	facilitate the entire calculation of C index.'''
				target_time_data = dataCat(target_time_data, target_batch[:, 1], i)
				target_event_data = dataCat(target_event_data, target_batch[:, 0], i)
				pred_hazard_data = dataCat(pred_hazard_data, out['hazard'].detach(), i)

				i += 1
				mean_loss += loss.item()
			'''Calculate C index'''
			time = target_time_data.cpu().numpy()
			event = target_event_data.cpu().numpy()
			haz_arr = pred_hazard_data.cpu().numpy()
			c_index_train = concordance_index(time, -haz_arr, event)

			print('Train C_index: %.4f' % c_index_train)
			print('Train mean Loss: %.4f' % (mean_loss/i))

			'''Test set'''
			net.eval()
			for val_step, (mrna_batch, cnv_batch, clin_cat_batch, clin_cont_batch, target_batch) in enumerate(validation_loader):
				out, _ = net(mrna=mrna_batch.to(device), cnv=cnv_batch.to(device), clin_cat=clin_cat_batch.to(device), clin_cont=clin_cont_batch.to(device))

				time = target_batch[:, 1].cpu().numpy()
				event = target_batch[:, 0].cpu().numpy()
				haz_arr = out['hazard'].detach().cpu().numpy()
				c_index_vali = concordance_index(time, -haz_arr, event)
				vali_c_index_arr[j, epoch] = c_index_vali
				print('Validation C_index: %.4f' % c_index_vali)
				print('--------------------------------------')

	mean_value = evaluate_model(vali_c_index_arr)
	print('The average C index is: ')
	print(mean_value)
	# print(vali_c_index_arr)

if __name__ == '__main__':
	main()




