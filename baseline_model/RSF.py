import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest

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

	# 5 Fold Corss Validation
	for k, (train, test) in enumerate(kfold):
		print(f'{modality}: Start {k+1} fold!')

		# train test split
		X_train, y_train = data.iloc[train, :].drop(['event', 'time'], axis=1), data.iloc[train, :][['event', 'time']]
		X_test, y_test = data.iloc[test, :].drop(['event', 'time'], axis=1), data.iloc[test, :][['event', 'time']]

		# create y array (event, time)
		y_train_arr = [(y_train['event'][i], y_train['time'][i]) for i in y_train.index.tolist()]
		y_train_arr = np.array(y_train_arr, dtype=[('cens', '?'), ('time', '<f8')])
		y_test_arr = [(y_test['event'][i], y_test['time'][i]) for i in y_test.index.tolist()]
		y_test_arr = np.array(y_test_arr, dtype=[('cens', '?'), ('time', '<f8')])

		# Random Survival Forest
		rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, max_features='sqrt', n_jobs=-1, random_state=SEED)
		rsf.fit(X_train, y_train_arr)
		scores.append(rsf.score(X_test, y_test_arr))

	m, s = eval_mean_std(scores)
	print('Mean: ', m)
	print('Std: ', s)

	with open(f'RSF_{modality}_score.txt', 'w') as file:
		file.write(str(scores))
		file.write(f'\n Mean: {m}')
		file.write(f'\n Std: {s}')
	file.close()





























