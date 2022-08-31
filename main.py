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

from module.model import *
from module.optim import *

parser = argparse.ArgumentParser()
parser.add_argument('--mode')
args = parser.parse_args()

file_name = input("Enter a PNS no : ")
tmp = open("tmp.txt", 'w')
tmp.write(file_name)
tmp.close()

exec(open("parser.py").read())
    
data = pd.read_csv(os.path.join('./', file_name, 'final_data.csv')).iloc[:, 1:]
tensor = data.to_numpy()
tensor = np.expand_dims(tensor, axis=0)

embedding_dim = 16
hidden_dim = 64
output_dim = 3
n_layers = 5
bidirectional = True
dropout_rate = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTM(embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate)
NGPU = torch.cuda.device_count()
if NGPU > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(NGPU)))
    # torch.multiprocessing.set_start_method('spawn', force=True)
model.module.load_state_dict(torch.load("./classificatior.pt"))
model = model.to(device)

model.module.load_state_dict(torch.load("./classificatior.pt"))

inference(model, tensor, args.mode, device, file_name)
