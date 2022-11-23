"""Keep 100 features with highest variance"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


mRNA_PATH = '/data/PanCancer_mRNA.txt'

# Read data
data_mrna = pd.read_csv(mRNA_PATH, sep='\t')
mrna_samples = pd.read_csv(mRNA_PATH, sep='\t', header=None, nrows=1)
mrna_samples = mrna_samples.values.tolist()
mrna_samples = mrna_samples[0]
mrna_samples = mrna_samples[1:]
for j in range(len(mrna_samples)):
	mrna_samples[j] = mrna_samples[j][:12]
data_mrna.columns = ['sample'] + mrna_samples
data_mrna.fillna(0.0, inplace=True)
data_mrna = data_mrna.T
data_mrna.drop(index='sample', inplace=True)


# Read patients ID in preprocessed clinical data
data_clin = pd.read_csv('/preprocessed_data/Pc_clinical_emb.csv', header=None)
clin_samples = data_clin[[0]]
clin_samples = clin_samples.values.tolist()
clinical_samples = list()
for i in range(len(clin_samples)):
	clinical_samples.append(clin_samples[i][0])
clin_samples = clinical_samples


# Remove the rows with same patient ID
data_mrna.reset_index(inplace=True)
data_mrna = data_mrna.drop_duplicates(['index'])
data_mrna.reset_index(drop=True)
data_mrna.set_index('index', inplace=True)


# Select the top 100 features with highest variance
var_arr = np.array(data_mrna.var())
var_list_idx = np.argsort(-var_arr)
top100idx = var_list_idx[:100]
data_mrna_100 = data_mrna.loc[:, top100idx]


# min-max normalization
scaler = MinMaxScaler()
mrna_0_1 = scaler.fit_transform(data_mrna_100)
mrna_f_df = pd.DataFrame(mrna_0_1)
mrna_f_df.index = data_mrna.index
mrna_f_df.reset_index(inplace=True)


# Create all zero vector
a = mrna_f_df[mrna_f_df['index'] == 'TCGA-OR-A5J1']
sample_row = a.copy()
sample_row['index'] = 'xx'
for i in range(mrna_f_df.shape[1]-1):
	sample_row[i] = 0.0


# Fill NaN with zero vectors
i = 0
for x in clin_samples:
	if i == 0:
		mrna = mrna_f_df[mrna_f_df['index']  == x]
		if mrna.shape[0] == 0:
			mrna = sample_row.copy()
			mrna['index'] = x
		i += 1
	else:
		mrna_row = mrna_f_df[mrna_f_df['index'] == x]
		if mrna_row.shape[0] == 0:
			mrna_row = sample_row.copy()
			mrna_row['index'] = x
		mrna = pd.concat([mrna, mrna_row], axis = 0)


# Save data
mrna.set_index('index', inplace=True)
mrna.to_csv(f'/preprocessed_data/PC_mRNA_100.csv', index=False, header=False)















