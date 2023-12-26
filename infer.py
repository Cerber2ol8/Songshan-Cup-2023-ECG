# Author: LCY
# update: 2023/10/25
import os
import shutil
import glob
import sys

import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from utils import MyDataset, process_label, convert_to_binary_labels, calculate_metrics

from model import create_model
from tqdm import tqdm

# data max: 157.5, min: -319.75
# data mean: 0.669921875, std: 0.007345963568431114

# 训练阶段得到的样本数据
mean = 0.669921875
std = 0.007345963568431114
_max = 157.5
_min = -319.75


def preprocess(raw_data, mean=mean, std=std, _min=_min, _max=_max):
    # 根据训练集的max和min 对输入数据进行截断 然后进行归一化
    raw_data = np.clip(raw_data, _min, _max)

    scaled_data = (raw_data - _min) / (_max - _min)
    processed_data = (raw_data - mean) / std
    
    return processed_data



def infer_one(model, inputs):
    model.eval()
    with torch.no_grad():
        inputs = preprocess(inputs).cuda()
        outpus = model(inputs)
        prob, pred = outpus.max(1, keepdim=True)
        pred = pred.view(1,).cpu()
        
    return pred

def load_data(data_dir):
    file_data_list = sorted(glob.glob("{}/*/*_data.npy".format(data_dir)))
    file_label_list = sorted(glob.glob("{}/*/*_label.npy".format(data_dir)))
    
    assert len(file_data_list)==len(file_label_list), "数据和标签数量不匹配！"
    
    data_list = []
    label_list = []
    for i in range(len(file_data_list)):
        x = np.load(file_data_list[i])
        y = np.load(file_label_list[i])
        
        x = x.reshape(x.shape[0]*x.shape[1],x.shape[2])
        y = process_label(y)
        data_list.append(x)
        label_list.append(y)
        
        
    data = np.concatenate(data_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    return data, label
    
    
    

def load_model(model_dir, model_id=0, n_classes=4, input_dim=1, hidden_dim=128, hidden_size=64):
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_id == 0:

        model = create_model('se-cnnsltm', in_dims=input_dim, hidden_dims=hidden_dim, hidden_size=hidden_size, num_classes=n_classes)
        model = model.to(device)
        model.load_state_dict(torch.load(model_dir))
        return model
    
    elif model_id == 1:
        #model = CNN_LSTM(in_dims=input_dim, hidden_dims=hidden_dim, hidden_size=hidden_size, num_classes=n_classes)
        model = create_model('se-resnet', block_num=[1,2,2,1], cnn_channels=[24,8,8,8], lstm_dims=[32,32,1], num_classes=n_classes, include_top=True)

        model = model.to(device)
        model.load_state_dict(torch.load(model_dir))
        return model


def cal_acc(model, data, label):
    correct = 0
    pred_label = []
    
    for i in tqdm(range(len(data)), desc="Processing", unit="iteration"):
        inputs = torch.tensor(data[i].astype(np.float32)).unsqueeze(0)
        pred = infer_one(model, inputs).cpu().numpy()
        pred_label.append(pred)
        if label[i]==pred[0]:
            correct += 1
    accuracy = correct / len(label)
    pred_label = np.array(pred_label)
    
    true_label = convert_to_binary_labels(label)
    pred_label = convert_to_binary_labels(pred_label)
    precision, recall, f1_score = calculate_metrics(true_label, pred_label)
    
    
    return accuracy, precision, recall, f1_score




if __name__ == '__main__':
    
    model = load_model("besk-ckpt.pt")

    data, label = load_data("data_raw")
    
    for inputs in data:
        inputs = torch.tensor(inputs.astype(np.float32)).unsqueeze(0)
        pred = infer_one(model, inputs).cpu()
        print(pred)
    
    
