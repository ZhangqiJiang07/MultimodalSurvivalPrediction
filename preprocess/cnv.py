"""CNV data preprocess"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

THRESHOLD = 0.2
CNV_PATH = '/data/PanCancer_CNV.txt'

def VarianceSelect(data, t):
	selector = VarianceThreshold(threshold=t)
	result_select = selector.fit_transform(data)
	result_support = selector.get_support(indices=True)
	return result_select, result_support

# Read CNV data
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
data_clin = pd.read_csv('/preprocessed_data/Pc_clinical_emb.csv', header=None)
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


# Variance threshold
res, _ = VarianceSelect(data_cnv, THRESHOLD)
cnv_df = pd.DataFrame(res)


# min-max normalization
scaler = MinMaxScaler()
cnv_0_1 = scaler.fit_transform(cnv_df)
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
cnv.to_csv(f'preprocessed_data/PC_CNV_threshold_{int(THRESHOLD*100)}.csv', header=False, index=False)










