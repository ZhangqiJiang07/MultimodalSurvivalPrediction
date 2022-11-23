""" Clinical data preprocess"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder


CLINICAL_PATH = '/data/PanCancer_Clinical.xlsx'

# Read clinical data
data = pd.read_excel(CLINICAL_PATH)


# Select the used features
data_used = data[['bcr_patient_barcode', 'type', 'gender', 'race', 'histological_type',
					'age_at_initial_pathologic_diagnosis', 'OS', 'OS.time']]
data_used.columns = ['id', 'cancer_type', 'gender', 'race', 'histological_type', 'age', 'event', 'event_time']


idx = data_used[data_used[['event', 'event_time']].isnull().T.any()].index
data_used.drop(labels=idx, inplace=True)
data_used.loc[data_used['race'] == '[Not Evaluated]', 'race'] = 'Na'
data_used.loc[data_used['race'] == '[Unknown]', 'race'] = 'Na'
data_used.loc[data_used['race'] == '[Not Available]', 'race'] = 'Na'


# categorical data and continuous data
data_id = data_used[['id']]
data_cate = data_used[['cancer_type', 'gender', 'race', 'histological_type']]
data_num = data_used[['age']]
target = data_used[['event', 'event_time']]


# Convert label to category type
data_cate['histological_type'] = data_cate['histological_type'].astype('str')
for col in data_cate.columns:
	data_cate[col] = LabelEncoder().fit_transform(data_cate[col])

for col in data_cate.columns:
	data_cate[col] = data_cate[col].astype('category')


# Use 0 to fill NaN in age column
idx_num = data_num[data_num.isnull().T.any()].index
data_num.loc[idx_num, 'age'] = 0


# Define the embedding size
embedded_cols = {n: len(col.cat.categories) for n,col in data_cate.items()}
embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _,n_categories in embedded_cols.items()]
print(embedding_sizes)


# concate the categorical data and continuous data
df = pd.concat([data_id, data_cate], 1)
df = pd.concat([df, data_num], 1)
df = pd.concat([df, target], 1)

# print('sample size: ', len(set(df['id'].values.tolist())))
# leng = [len(i) for i in df['id'].values.tolist()]
# print(set(leng))

#Save data
df.to_csv('/preprocessed_data/Pc_clinical_emb.csv', header=False, index=False)







