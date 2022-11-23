import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index
from lifelines.utils.sklearn_adapter import sklearn_adapter
from sklearn.model_selection import train_test_split, GridSearchCV


CLIN_PATH = 'Pc_clinical_emb.csv'
MIRNA_PATH = 'PC_miRNA_100.csv'
MRNA_PATH = 'PC_mRNA_100.csv'
CNV_PATH = 'PC_CNV_100.csv'
SEED = 24
K = 5


def eval_mean_std(scoreList):
	m = np.sum(scoreList, axis=0) / len(scoreList)
	s = np.std(scoreList)

	return m, s


data_clin = pd.read_csv(CLIN_PATH, header=None)
data_clin.columns = ['ID', 'cancer_type', 'gender', 'race', 'histological_type', 'age', 'event', 'time']
data_clin.drop(['ID'], axis=1, inplace=True)


modalities = ['clinical', 'mrna', 'mirna', 'cnv']
data_paths = [CLIN_PATH, MRNA_PATH, MIRNA_PATH, CNV_PATH]

for modality, data_path in zip(modalities, data_paths):
	if modality != 'clinical':
		# Read Data
		data_df = pd.read_csv(data_path, header=None)
		data = pd.concat([data_df, data_clin[['event', 'time']]], axis=1)
	else:
		data = data_clin
	# 5 fold
	X = data.drop(['event', 'time'], axis=1)
	y = data.event

	strtfdKFold = StratifiedKFold(n_splits=K, random_state=SEED, shuffle=True)
	kfold = strtfdKFold.split(X, y)
	scores = []

	for k, (train, test) in enumerate(kfold):
		print(f'{modality}: Start {k+1} fold!')
		train_set, test_set = data.iloc[train, :], data.iloc[test, :]
		cph = CoxPHFitter(penalizer=0.01)
		cph.fit(train_set, duration_col='time', event_col='event')
		scores.append(concordance_index(test_set['time'], -cph.predict_partial_hazard(test_set), test_set['event']))

	m, s = eval_mean_std(scores)
	print('Mean: ', m)
	print('Std: ', s)

	with open(f'CPH_{modality}_score.txt', 'w') as file:
		file.write(str(scores))
		file.write(f'\n Mean: {m}')
		file.write(f'\n Std: {s}')
	file.close()








