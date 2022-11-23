"""Pancancer survival prediction using a deep learning architecture with multimodal representation and integration"""

# import sys
# sys.path.append('/sub_models')

import torch
import torch.nn as nn
import torch.nn.functional as F

from sub_models.mirnaNet import MirnaNet
from sub_models.mrnaNet import MrnaNet
from sub_models.cnvNet import CnvNet
from sub_models.clinicalEmbeddingNet import ClinicalEmbeddingNet
from sub_models.attention import Attention

class Net(nn.Module):
	def __init__(self, modalities, m_length, device=None,
					input_modality_dim={'clinical':4, 'mRNA':1579, 'miRNA':743, 'CNV':2711}):
		"""
		Parameters
		----------
		modalities: list
			Used modalities.

		m_length: int
			Representation length.

		"""
		super(Net, self).__init__()
		self.data_modalities = modalities
		self.m_length = m_length
		self.dim = input_modality_dim
		self.device = device

		self.submodel_pipeline = {}
		# clinical -----------------------------------------------#
		if 'clinical' in self.data_modalities:
			self.clinical_submodel = ClinicalEmbeddingNet(m_length=self.m_length)
			self.submodel_pipeline['clinical'] = self.clinical_submodel

		# mRNA ---------------------------------------------------#
		if 'mRNA' in self.data_modalities:
			self.mRNA_submodel = MrnaNet(mrna_length=self.dim['mRNA'], m_length=self.m_length)
			self.submodel_pipeline['mRNA'] = self.mRNA_submodel

		# miRNA --------------------------------------------------#
		if 'miRNA' in self.data_modalities:
			self.miRNA_submodel = MirnaNet(mirna_length=self.dim['miRNA'], m_length=self.m_length)
			self.submodel_pipeline['miRNA'] = self.miRNA_submodel

		# CNV ----------------------------------------------------#
		if 'CNV' in self.data_modalities:
			self.CNV_submodel = CnvNet(cnv_length=self.dim['CNV'], m_length=self.m_length)
			self.submodel_pipeline['CNV'] = self.CNV_submodel

		# Fusion -------------------------------------------------#
		if len(self.data_modalities) > 1:
			self.fusion = Attention(m_length=self.m_length, device=self.device)


		# Survival prediction
		self.hazard_layer1 = nn.Linear(m_length, 256)
		self.hazard_layer2 = nn.Linear(256, 128)
		self.hazard_layer3 = nn.Linear(128, 64)
		self.hazard_layer4 = nn.Linear(64, 1)


	def forward(self, x):
		"""
		Parameters
		----------
		x: dictionary
			Input data from different modality, like:
			{'clinical': tensor, 'mRNA':tensor, }
		"""
		# Extract representations from different modality
		representation = {}
		flag = 1
		for modality in x:
			if modality in ['clinical_categorical', 'clinical_continuous']:
				if flag:
					representation['clinical'] = self.submodel_pipeline['clinical'](x['clinical_categorical'], x['clinical_continuous'])
					flag = 0
				continue
			representation[modality] = self.submodel_pipeline[modality](x[modality])
		# fusion part
		if len(self.data_modalities) > 1:
			x = self.fusion(representation)
		else:
			x = representation[self.data_modalities[0]]

		# survival predict
		hazard = self.hazard_layer1(x)
		hazard = self.hazard_layer2(hazard)
		hazard = self.hazard_layer3(hazard)
		hazard = torch.sigmoid(self.hazard_layer4(hazard))

		return {'hazard':hazard}, representation







































