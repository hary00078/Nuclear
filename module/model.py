import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import sys
import os
import json

import argparse
from module.visualize import *
import pickle

class Attention(nn.Module):
    
    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):

        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)
        mix = torch.bmm(attention_weights, context)


        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)
        
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights

class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate):
        
        super().__init__()
        self.embed = nn.Embedding(5, hidden_dim*2)
        self.lstm_list =nn.ModuleList([nn.LSTM(1, hidden_dim, n_layers, bidirectional=bidirectional,
                            dropout=dropout_rate, batch_first=True) for _ in range(16)])
        self.att = Attention(hidden_dim*2)
        self.fc = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def execute_lstm(self, X):
        ch_num = X.shape[-1]
        hidden_list = []
        
        for i in range(ch_num):
            channel = X[:, :, i].unsqueeze(-1)
            lstm = self.lstm_list[i]
            hidden, _ = lstm(channel)
            logit = hidden[:, -1, :]
            hidden_list.append(logit)
            
        return torch.stack(hidden_list, dim=1)
            
    
    def forward(self, X):
        lstm_out = self.execute_lstm(X)
        m = torch.tensor([[0, 1, 2, 3, 4]]).expand(X.shape[0], -1).to(X.device)
        m_embed = self.embed(m)
        att_out, _ = self.att(m_embed, lstm_out)
        att_out = torch.sum(att_out, dim=1)
        predict = self.fc(att_out)

        return predict