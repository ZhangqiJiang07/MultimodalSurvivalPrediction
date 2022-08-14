'''
1+1型 : Clinical AND (miRNA OR mRNA OR CNV)
'''

import pandas as pd
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def Split2VaTSampler(lenth):
    indices = list(range(lenth))
    split = int(np.floor(0.2*lenth))

    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)
    return train_sampler, validation_sampler

def FindOutIndex(data):
    out_idx = []
    target = data.values.tolist()
    for i in range(data.shape[0]):
        try:
            target_ex = np.array(target[i], dtype='int')
        except:
            out_idx.append(i)
    return out_idx

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path_data1, path_clin):
        data = pd.read_csv(path_clin, header=None)
        target_data = data[[6, 7]]
        out_idx = FindOutIndex(target_data)
        data.drop(index=out_idx, inplace=True)
        # data.reset_index(drop=True)

        a = data[[1, 2, 3, 4, 5]]
        idx = a[a[5].isnull()].index
        g = [i for i in idx]
        data.drop(index=g, inplace=True)
        # data.reset_index(drop=True)

        target_data = data[[6, 7]]
        self.target = target_data.values.tolist()
        clin_data_cat = data[[1, 2, 3, 4]]
        clin_data_cont = data[[5]]
        self.clin_cat = clin_data_cat.values.tolist()
        self.clin_cont = clin_data_cont.values.tolist()

        data1 = pd.read_csv(path_data1, header=None)
        data1.drop(index=out_idx + g, inplace=True)
        # data1.reset_index(drop=True)
        self.data1 = data1.values.tolist()

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        #Clinical Categorical
        clin_cat = np.array(self.clin_cat[index]).astype(np.int64)
        clin_cat = torch.from_numpy(clin_cat)
        #Clinical Continuous
        clin_cont = np.array(self.clin_cont[index]).astype(np.float32)
        clin_cont = torch.from_numpy(clin_cont)

        data_m1 = np.array(self.data1[index])
        data_m1 = torch.from_numpy(data_m1)
        data_m1 = data_m1.type(torch.FloatTensor)

        target_y = np.array(self.target[index], dtype='int')
        target_y = torch.from_numpy(target_y)
        target_y = target_y.type(torch.LongTensor)
        return data_m1, clin_cat, clin_cont, target_y















