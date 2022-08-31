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


def sampling(data, device):
    length = data.shape[1]
    cat = []
    curr_idx = 0
    while True:
        if curr_idx + 50 > length:
            break
        curr_tensor = data[:, curr_idx: curr_idx+50, :]
        torch_tensor = torch.FloatTensor(curr_tensor).to(device)
        cat.append(torch_tensor)
        curr_idx += 1
    concat_tensor = torch.cat(cat, dim=0)
    return concat_tensor

def save_result_json(data, error_net, file_name):
    f = open(os.path.join('./', file_name, 'defect_location.txt'), 'r')
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
    return def_start, def_end
    
def make_error_net(error_candi):
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
    revised_error_net = [error_net[0]]
    for i in range(1, len(error_net)):
        if len(set(range(*error_net[i])) & set(range(*error_net[i-1]))) == 0 :
            revised_error_net.append(error_net[i])
    return revised_error_net
    
def inference(model, data, mode, device, file_name):
    model.eval()
    pred_list = []
    prob_list = []
    error_candi = []

    if mode == 'all':
        print("------------------------------")
        print("--------Inference Error-------")
        print("------------------------------")
        with torch.no_grad():
            concat_tensor = sampling(data, device)
            loader = DataLoader(concat_tensor, batch_size=2048, shuffle=False)
            for idx, batch in enumerate(loader):
                logit = model(batch)
                soft_logit = F.softmax(logit, dim=-1).squeeze()
                argmax_logit = torch.argmax(soft_logit, dim=-1)
                pred_list += argmax_logit.reshape(-1).detach().cpu().tolist()
                
        for idx, i in enumerate(pred_list):
            if i == 2:
                error_candi.append([idx, idx + 50])
        
        error_net = make_error_net(error_candi)
        def_start, def_end = save_result_json(data, error_net, file_name)
            
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
    return