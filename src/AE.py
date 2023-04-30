#!/usr/bin/env python

import sys
import torch
from torch import nn
import torch.nn.functional as F

###Define an Autoencoder NetWork
class Auto_Exp(nn.Module):
    def __init__(self,features_num):
        super(Auto_Exp,self).__init__()
        self.features_num = features_num

        # def the encoder function
        self.enc1 = nn.Linear(in_features=self.features_num, out_features=2048)
        self.enc2 = nn.Linear(in_features=2048, out_features=256)
        self.enc3 = nn.Linear(in_features=256, out_features=32)
        

        # def the decoder function
        self.dec1 = nn.Linear(in_features=32, out_features=256)
        self.dec2 = nn.Linear(in_features=256, out_features=2048)
        self.dec3 = nn.Linear(in_features=2048, out_features=self.features_num)
        
    # def forward function
    def forward(self, x):
        # encoding
        x = F.leaky_relu(self.enc1(x))
        x = F.leaky_relu(self.enc2(x))
        x = F.leaky_relu(self.enc3(x))
        y = x

        # decoding
        x = F.leaky_relu(self.dec1(x))
        x = F.leaky_relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        return x,y
        
        
###Define an Autoencoder NetWork
class Auto_PathL_Exp(nn.Module):
    def __init__(self,features_num):
        super(Auto_Path_Exp,self).__init__()
        self.features_num = features_num

        # def the encoder function
        self.enc1 = nn.Linear(in_features=self.features_num, out_features=64)
        self.enc2 = nn.Linear(in_features=64, out_features=8)
        
        # def the decoder function
        self.dec1 = nn.Linear(in_features=8, out_features=64)
        self.dec2 = nn.Linear(in_features=64, out_features=self.features_num)
        
    # def forward function
    def forward(self, x):
        # encoding
        x = F.leaky_relu(self.enc1(x))
        x = F.leaky_relu(self.enc2(x))
        y = x

        # decoding
        x = F.leaky_relu(self.dec1(x))
        x = torch.sigmoid(self.dec2(x))
        return x,y

###Define an Autoencoder NetWork
class Auto_PathM_Exp(nn.Module):
    def __init__(self,features_num):
        super(Auto_Path_Exp,self).__init__()
        self.features_num = features_num

        # def the encoder function
        self.enc1 = nn.Linear(in_features=self.features_num, out_features=8)
        
        # def the decoder function
        self.dec1 = nn.Linear(in_features=8, out_features=self.features_num)
        
    # def forward function
    def forward(self, x):
        # encoding
        x = F.leaky_relu(self.enc1(x))
        y = x

        # decoding
        x = torch.sigmoid(self.dec1(x))
        return x,y