"""Keep 100 features with highest variance"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


CNV_PATH = '/data/PanCancer_CNV.txt'

# Read data
data_cnv = pd.read_csv(CNV_PATH, sep='\t')
cnv_samples = pd.read_csv(CNV_PATH, sep='\t', header=None, nrows=1)
cnv_samples = cnv_samples.values.tolist()
cnv_samples = cnv_samples[0]
cnv_samples = cnv_samples[1:]
for j in range(len(cnv_samples)):
	cnv_samples[j] = cnv_samples[j][:12]
data_cnv.columns = ['sample'] + cnv_samples
data_cnv.fillna(0.0, inplace=True)
data_cnv = data_cnv.T
data_cnv.drop(index='sample', inplace=True)


# Read patients ID in preprocessed clinical data
data_clin = pd.read_csv('preprocessed_data/Pc_clinical_emb.csv', header=None)
clin_samples = data_clin[[0]]
clin_samples = clin_samples.values.tolist()
clinical_samples = list()
for i in range(len(clin_samples)):
	clinical_samples.append(clin_samples[i][0])
clin_samples = clinical_samples


# Remove the rows with same patient ID
data_cnv.reset_index(inplace=True)
data_cnv = data_cnv.drop_duplicates(['index'])
data_cnv.reset_index(drop=True)
data_cnv.set_index('index', inplace=True)


# Select the top 100 features with highest variance
var_arr = np.array(data_cnv.var())
var_list_idx = np.argsort(-var_arr)
top100idx = var_list_idx[:100]
data_cnv_100 = data_cnv.loc[:, top100idx]


# min-max normalization
scaler = MinMaxScaler()
cnv_0_1 = scaler.fit_transform(data_cnv_100)
cnv_f_df = pd.DataFrame(cnv_0_1)
cnv_f_df.index = data_cnv.index
cnv_f_df.reset_index(inplace=True)


# Create all zero vector
a = cnv_f_df[cnv_f_df['index'] == 'TCGA-A5-A0GI']
sample_row = a.copy()
sample_row['index'] = 'xx'
for j in range(cnv_f_df.shape[1]-1):
	sample_row[j] = 0.0


# Fill NaN with zero vectors
i = 0
for x in clin_samples:
	if i == 0:
		cnv = cnv_f_df[cnv_f_df['index']  == x]
		if cnv.shape[0] == 0:
			cnv = sample_row.copy()
			cnv['index'] = x
		i += 1
	else:
		cnv_row = cnv_f_df[cnv_f_df['index'] == x]
		if cnv_row.shape[0] == 0:
			cnv_row = sample_row.copy()
			cnv_row['index'] = x
		cnv = pd.concat([cnv, cnv_row], axis = 0)


# Save data
cnv.set_index('index', inplace=True)
cnv.to_csv(f'/preprocessed_data/PC_CNV_100.csv', header=False, index=False)







