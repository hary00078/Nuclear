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

def inference(model, data, stride, mode, file_name=file_name):
    model.eval()
    length = data.shape[1]
    pred_list = []
    prob_list = []
    error_candi = []
    curr_idx=0
    
    if mode == 'all':
        print("------------------------------")
        print("--------Inference Error-------")
        print("------------------------------")
        with torch.no_grad():
            cat = []
            while True:
                if curr_idx + 50 > length:
                    break
                curr_tensor = data[:, curr_idx: curr_idx+50, :]
                torch_tensor = torch.FloatTensor(curr_tensor).to(device)
                cat.append(torch_tensor)
                curr_idx += stride
            concat_tensor = torch.cat(cat, dim=0)
            loader = DataLoader(concat_tensor, batch_size=2048, shuffle=False)
            for idx, batch in enumerate(loader):
                logit = model(batch)
                soft_logit = F.softmax(logit, dim=-1).squeeze()
                argmax_logit = torch.argmax(soft_logit, dim=-1)
                pred_list += argmax_logit.reshape(-1).detach().cpu().tolist()
                
        for idx, i in enumerate(pred_list):
            if i == 2:
                error_candi.append([idx*stride, idx*stride + 50])
        
        error_net = []
        history = 0
        for i in range(len(error_candi)-1):
            curr_start = error_candi[i][0]
            curr_end = error_candi[i][1]
            next_start = error_candi[i+1][0]
            next_end = error_candi[i+1][1]
            
            if curr_start == next_start-1:
                history += 1
            else:
                save_end = curr_end
                save_start = error_candi[i-history][0]
                error_net.append([save_start, save_end])
                history = 0
                
        f = open(os.path.join('./', file_name, 'defect_location.txt'))
        def_loc = f.readline().split('\t')
        def_start = int(def_loc[0])
        def_end = int(def_loc[1])
        f.close()
        
        save_dict = {}
        save_dict["final_data"] = data.tolist()
        save_dict["defect_tool"] = [def_start, def_end]
        save_dict["defect_ai"] = error_net
        
        json_path = os.path.join('./', file_name, 'information.json')
        with open(json_path, 'w') as f:
            json.dump(save_dict, f)
            
        visual = Visualize(file_name, [def_start, def_end], error_net)
        visual.extract_feature()
        # visual.visulalize_defect()
        error_net = visual.visulalize_report()
        if len(error_net) != 0:
            for idx, item in enumerate(error_net):
                item_start = item[0]
                item_end = item[1]
                item_tensor = data[:, item_start:item_end, :].tolist()
                open_file = open(os.path.join('./', file_name, 'picture','ai', str(idx+1), 'defect_tensor.pkl'),  "wb")
                pickle.dump(item_tensor, open_file)
                open_file.close()
    
    elif mode == 'check':
        f = open(os.path.join('./', file_name, 'defect_location.txt'))
        def_loc = f.readline().split('\t')
        f.close()
        start = 4480
        end = 4537
        with torch.no_grad():
            for i in range(50):
                curr_tensor = data[:, start:end, :]
                torch_tensor = torch.FloatTensor(curr_tensor).to(device)
                logit = model(torch_tensor)
                soft_logit = F.softmax(logit, dim=-1).squeeze()
                print(soft_logit)
            
        return soft_logit
        
stride = 1
result = inference(model, tensor, stride, args.mode)
