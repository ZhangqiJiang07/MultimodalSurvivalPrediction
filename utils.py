import torch
import numpy as np

#define utils
def dataCat(data1, data2, i):
    if i == 0:
        data1 = data2
    else:
        data1 = torch.cat([data1, data2], dim=0)
    return data1

def test_gpu():
    print('GPU？')
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'The device is {device}!')
    # print(device)
    return device

# def create_loader(path1, path2, batch_size):
#     print('Creating Dataset...')
#     my_data_set = MyDataset(path1, path2)
#     train_sampler, validation_sampler = Split2VaTSampler(my_data_set.__len__())
#     train = DataLoader(my_data_set, batch_size=batch_size, sampler=train_sampler)
#     validation = DataLoader(my_data_set, batch_size=len(validation_sampler), sampler=validation_sampler)
#     return my_data_set, train, validation

def split_to_KFold(length, k, ratio=0.2):
    indices = list(range(length))
    split = int(np.floor(ratio*length))
    folds = []
    for i in range(k):
        if i < k-1:
            fold = np.random.choice(indices, size=split, replace=False)
            indices = list(set(indices) - set(fold))
            folds.append(fold)
        else:
            folds.append(np.array(indices))
    return folds

def evaluate_model(mat):
    value = 0
    for row in range(mat.shape[0]):
        value += max(mat[row, :])
    final_value = value / mat.shape[0]
    return final_value