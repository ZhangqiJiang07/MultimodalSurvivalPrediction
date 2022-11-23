"""miRNA data preprocess"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

MIRNA_PATH = '/data/PanCancer_mirna.txt'


# Read miRNA data
data_mirna = pd.read_csv(MIRNA_PATH, sep='\t')
mirna_samples = pd.read_csv(MIRNA_PATH, sep='\t', header=None, nrows=1)
mirna_samples = mirna_samples.values.tolist()
mirna_samples = mirna_samples[0]
mirna_samples = mirna_samples[1:]
for j in range(len(mirna_samples)):
	mirna_samples[j] = mirna_samples[j][:12]
data_mirna.columns = ['sample'] + mirna_samples
data_mirna.fillna(0.0, inplace=True)
data_mirna = data_mirna.T
data_mirna.drop(index='sample', inplace=True)


# Read patients ID in preprocessed clinical data
data_clin = pd.read_csv('/preprocessed_data/Pc_clinical_emb.csv', header=None)
clin_samples = data_clin[[0]]
clin_samples = clin_samples.values.tolist()
clinical_samples = list()
for i in range(len(clin_samples)):
	clinical_samples.append(clin_samples[i][0])
clin_samples = clinical_samples


# Remove the rows with same patient ID
data_mirna.reset_index(inplace=True)
data_mirna = data_mirna.drop_duplicates(['index'])
data_mirna.reset_index(drop=True)
data_mirna.set_index('index', inplace=True)


# min-max normalization
scaler = MinMaxScaler()
mirna_0_1 = scaler.fit_transform(data_mirna)
mirna_f_df = pd.DataFrame(mirna_0_1)
mirna_f_df.index = data_mirna.index
mirna_f_df.reset_index(inplace=True)


# Create all zero vector
a = mirna_f_df[mirna_f_df['index'] == 'TCGA-AG-3586']
sample_row = a.copy()
sample_row['index'] = 'xx'
for j in range(mirna_f_df.shape[1]-1):
	sample_row[j] = 0.0


# Fill NaN with zero vectors
i = 0
for x in clin_samples:
	if i == 0:
		mirna = mirna_f_df[mirna_f_df['index']  == x]
		if mirna.shape[0] == 0:
			mirna = sample_row.copy()
			mirna['index'] = x
		i += 1
	else:
		mirna_row = mirna_f_df[mirna_f_df['index'] == x]
		if mirna_row.shape[0] == 0:
			mirna_row = sample_row.copy()
			mirna_row['index'] = x
		mirna = pd.concat([mirna, mirna_row], axis = 0)

# Save data
mirna.set_index('index', inplace=True)
mirna.to_csv(f'/preprocessed_data/PC_miRNA.csv', header=False, index=False)










