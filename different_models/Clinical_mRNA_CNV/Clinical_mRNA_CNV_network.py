'''
Clinical + mRNA + CNV
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ClinicalEmbeddingNet import *
from itertools import *


def atten_matmul(mat1, mat2, n, m):
    i = 0
    for mat in mat2:
        if i == 0:
            final_mat = torch.bmm(mat1, mat).view(1, n, m, 1)
            i+=1
        else:
            final_mat = torch.cat([final_mat, torch.bmm(mat1, mat).view(1, n, m, 1)], dim=0)
    return final_mat

def atten_fusion3(info1, info2, info3):
    sample_num = info1.shape[0]
    info_lenth = info1.shape[1]
    a = info1.view(sample_num, 1, info_lenth, 1)
    b = info2.view(sample_num, 1, info_lenth, 1)
    c = info3.view(sample_num, 1, info_lenth, 1)
    fused_info = torch.cat([a, b, c], dim=1)
    return fused_info

def split_contrastive_data(batch_size):
    indices = list(range(batch_size))
    split_size = int(batch_size*0.5)
    np.random.shuffle(indices)
    x1, x2 = indices[:split_size], indices[split_size:]
    if len(x1) != len(x2):
        _ = x2.pop()
    return x1, x2

def contrastive_loss(x1, x2, v, label, margin=0.1):
    loss = 0
    for i in range(len(x1)):
        loss_x_x = 0
        dis_x_x = 0
        dis_y_y = 0
        for j in range(v[x1[i]].shape[0]-1):
            for k in range(j+1, v[x1[i]].shape[0]):
                dis_x_x += torch.cosine_similarity(v[x1[i]][j], v[x1[i]][k], dim=0)[0]
                dis_y_y += torch.cosine_similarity(v[x2[i]][j], v[x2[i]][k], dim=0)[0]

        dis = torch.cosine_similarity(v[x1[i]], v[x2[i]], dim=1).sum()
        loss += ((1-label)*torch.pow(dis, 2) + label*torch.pow(torch.clamp(margin-dis+0.5*dis_x_x+0.5*dis_y_y, min=0.0), 2))
    return loss

def circular_loss(batch_size, v, times):
    i = 0
    mean_loss = 0
    while i < times:
        i += 1
        x1, x2 = split_contrastive_data(batch_size)
        mean_loss += contrastive_loss(x1, x2, v, 1)
    return mean_loss/times

class AtteNet(nn.Module):

    def __init__(self, mrna_lenth, cnv_lenth, embedding_sizes, n_cont, n_model, m_lenth):
        super(AtteNet, self).__init__()
        self.mrna_lenth = mrna_lenth
        self.cnv_lenth = cnv_lenth
        self.n_model = n_model
        self.m_lenth = m_lenth
        self.embedding_sizes = embedding_sizes
        self.n_cont = n_cont

        self.W = nn.Parameter(torch.rand(n_model, m_lenth, m_lenth))
        # clin Sub-Model
        self.clin_embedding_net = CategoricalEmbedding(self.embedding_sizes, self.n_cont, self.m_lenth)
        # mrna Sub-Model
        self.mrna_hidden1 = nn.Linear(mrna_lenth, 800)
        self.mrna_hidden2 = nn.Linear(800, 500)
        self.mrna_hidden3 = nn.Linear(500, 200)
        self.mrna_hidden4 = nn.Linear(200, m_lenth)
        # cnv Sub-Model
        self.cnv_hidden1 = nn.Linear(cnv_lenth, 1200)
        self.cnv_hidden2 = nn.Linear(1200, 800)
        self.cnv_hidden3 = nn.Linear(800, 400)
        self.cnv_hidden4 = nn.Linear(400, 200)
        self.cnv_hidden5 = nn.Linear(200, m_lenth)

        #Batch Normalization
        self.mrna_bn1 = nn.BatchNorm1d(800)
        self.mrna_bn2 = nn.BatchNorm1d(500)
        self.mrna_bn3 = nn.BatchNorm1d(200)
        self.mrna_bn4 = nn.BatchNorm1d(128)
        
        self.cnv_bn1 = nn.BatchNorm1d(1200)
        self.cnv_bn2 = nn.BatchNorm1d(800)
        self.cnv_bn3 = nn.BatchNorm1d(400)
        self.cnv_bn4 = nn.BatchNorm1d(200)
        self.cnv_bn5 = nn.BatchNorm1d(128)

        #output可以考虑增加层数
        self.hazard_output1 = nn.Linear(m_lenth, 1)

        #Dropout Layer
        self.dropout_layer1 = nn.Dropout(p=0.3)
        self.dropout_layer2 = nn.Dropout(p=0.3)

    def forward(self, mrna, cnv, clin_cat, clin_cont):
        mrna = torch.relu(self.mrna_bn1(self.mrna_hidden1(mrna)))
        mrna = torch.relu(self.mrna_bn2(self.mrna_hidden2(mrna)))
        mrna = torch.relu(self.mrna_bn3(self.mrna_hidden3(mrna)))
        mrna = torch.relu(self.mrna_bn4(self.mrna_hidden4(mrna)))

        cnv = torch.relu(self.cnv_bn1(self.cnv_hidden1(cnv)))
        cnv = torch.relu(self.cnv_bn2(self.cnv_hidden2(cnv)))
        cnv = torch.relu(self.cnv_bn3(self.cnv_hidden3(cnv)))
        cnv = torch.relu(self.cnv_bn4(self.cnv_hidden4(cnv)))
        cnv = torch.relu(self.cnv_bn5(self.cnv_hidden5(cnv)))

        clin = self.clin_embedding_net(clin_cat, clin_cont)
        samples = mrna.shape[0]
        mrna = self.dropout_layer1(mrna)
        cnv = self.dropout_layer2(cnv)

        v = atten_fusion3(mrna, cnv, clin)
        attn_matrix = atten_matmul(self.W, v, self.n_model, self.m_lenth)
        attn_matrix = F.softmax(torch.tanh(attn_matrix), dim=1)
        c = torch.mul(v, attn_matrix)
        c = c.view(samples, self.n_model, self.m_lenth)
        c = c.sum(dim=1)

        hazard = self.hazard_output1(c)

        return {"hazard": hazard}, v

    def loss(self, pred, target, v, t):
        risk = pred['hazard']
        E = target[:, 0]
        hazard_ratio = torch.exp(risk)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = risk.T - log_risk
        censored_likelihood = uncensored_likelihood * E
        num_observed_events = torch.sum(E)
        neg_likelihood = -torch.sum(censored_likelihood) / num_observed_events

        contrast_loss = circular_loss(v.shape[0], v, t)

        return neg_likelihood + contrast_loss





























