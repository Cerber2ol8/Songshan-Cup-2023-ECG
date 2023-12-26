# Author: LCY
# update: 2023/10/25
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import glob
import shutil


# 处理标签数据
def process_label_bin(array):
    # 二分类标签

    char_array = np.array(list(array[0]))
    # 创建布尔掩码，'A'的位置为True，其他位置为False
    mask = (char_array == 'A')
    # 将非'A'值替换为0
    label = np.where(mask, 1, 0).astype(np.float32)
    # reshape 
    label = label.reshape(1, len(label))
    return label


def process_label(array):
    # 多分类标签 one-hot 编码

    array = np.array(list(array[0]))
    
    
    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    
    label_array = np.array([mapping[char] for char in array])
    label = label_array.reshape(1, len(label_array))
    
    return label_array


class MyDataset(Dataset):
    def __init__(self, data_dir, mean, std):
        self.data_dir = data_dir
        self.data_file = os.path.join(self.data_dir, 'data.npy')     
        assert os.path.exists(self.data_file), f'data file {self.data_file} not exists!'
        self.data = np.load(self.data_file)
        self.mean = mean
        self.std = std
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        x = sample[:-1]  # 数据部分
        y = sample[-1]   # 标签部分
        # print(sample.shape,x,y)

        data = torch.tensor(x, dtype=torch.float32)
        label = torch.tensor(y, dtype=torch.float32)
        
        #data = (data - self.mean) / self.std


        return data, label
    
    
    
def mse_ce_loss(data_rec, data, out_labels, labels):
    """
    计算重构和分类损失的函数
    :param data_rec:重建的信号
    :参数数据:实际的信号
    :param out_labels:预测的标签
    :param labels:实际标签
    :返回:重建的MSE损失和分类的CE损
    """
    #mse_loss = torch.nn.MSELoss(reduction='sum')(data_rec, data)
    mse_loss = torch.nn.L1Loss(reduction='sum')(data_rec, data)
    ce_loss = torch.torch.nn.CrossEntropyLoss(reduction='sum')(out_labels, labels)
    

    return mse_loss, ce_loss




def normal(data_array):
    _max = data_array.max()
    _min = data_array.min()
    #scaled_data = (data_array - _min) / (_max - _min)
    mean = scaled_data.mean()
    std = scaled_data.astype(np.float64).std()
    print(f"data max: {_max}, min: {_min}")
    print(f"data min: {mean}, std: {std}")
    processed_data = (data_array - mean) / std
    return processed_data
    
    

# 将数据复制到另一个目录并做区分处理
def process_data(args, is_train=False):
    
    file_data = sorted(glob.glob("{}/*/*_data.npy".format(args.raw_data)))
    file_label = sorted(glob.glob("{}/*/*_label.npy".format(args.raw_data)))
    assert len(file_data)==len(file_label), 'number of data not equal to number of label! '
    
    tmp_dir = 'tmp_data'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    for folder in os.listdir(args.raw_data):
        if folder != ".DS_Store":
            for file in os.listdir(os.path.join(args.raw_data, folder)):
                if file != ".DS_Store":
                    # 数据是用MacOS生成的 该文件会影响文件复制
                    shutil.copyfile(os.path.join(args.raw_data, folder,file),os.path.join(tmp_dir, file))

    data_list = glob.glob(f'{tmp_dir}/*_data.npy')
    ids = []
    for file_path in data_list:
        file_name = file_path.split('/')[-1]
        id = file_name.split('_')[0]
        ids.append(id)

    for id in ids:
        # 检查是否有缺失
        assert os.path.exists(f'data/{id}_data.npy'),f'label {id} does not exists!'
    print(f'total data :{len(ids)}')

    # 将所有数据整合成

    data_list = []
    label_list = []

    # 读取每个.npy文件并将其添加到 array_list
    for idx in ids:
        data = np.load(f'data/{idx}_data.npy')
        label = process_label(np.load(f'data/{idx}_label.npy'))
        data_list.append(data)
        label_list.append(label)


    data_array = np.array(data_list)
    label_array = np.array(label_list)
    data_array = data_array.reshape((data_array.shape[0] * data_array.shape[1] * data_array.shape[2],  data_array.shape[3]))
    label_array = label_array.reshape((label_array.shape[0] * label_array.shape[1] , 1))
    
    dataset = np.concatenate((data_array, label_array[:, :, np.newaxis]), axis=-1)
    dataset = dataset.reshape(dataset.shape[0]*dataset.shape[1],dataset.shape[2])

    print('data: ', data_array.shape)
    print('label: ', label_array.shape)

    if is_train:
        # 进行归一化
        data = norm(data)

    # 合并数据和标签
    dataset = np.concatenate((processed_data, label_array), axis=-1)
    print('dataset: ', dataset.shape)
    
    
    try:
        shutil.rmtree(args.data_dir)
    except Exception as e:
        pass
    os.mkdir(args.data_dir)
    
    
    if is_train:
        # 随机划分训练集和验证集
        from sklearn.model_selection import train_test_split
        train_ratio = 0.7
        val_ratio = 0.3
        train_data, val_data = train_test_split(dataset, train_size=train_ratio, test_size=val_ratio, random_state=42)

        print("train data:", train_data.shape)
        print("val data:", val_data.shape)
        
        os.mkdir(os.path.join(args.data_dir,"train"))
        os.mkdir(os.path.join(args.data_dir,"val"))
        np.save(f'{args.data_dir}/train/data.npy',train_data)
        np.save(f'{args.data_dir}/val/data.npy',val_data)
    else:
        os.mkdir(os.path.join(args.data_dir,"test"))
        np.save(f'{args.data_dir}/test/data.npy',dataset)
        
        
    try:
        shutil.rmtree(tmp_dir)
    except Exception as e:
        pass
    
    
def convert_to_binary_labels(original_labels):
    # 将多分类结果转化为二分类
    binary_labels = []
    for label in original_labels:
        if label == 0:
            binary_labels.append(0)  # 类别A
        else:
            binary_labels.append(1)  # 类别B
    return binary_labels

def calculate_metrics(true_labels, predicted_labels):
    # 初始化各种统计变量
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label == 1 and predicted_label == 1:
            true_positives += 1
        elif true_label == 0 and predicted_label == 1:
            false_positives += 1
        elif true_label == 1 and predicted_label == 0:
            false_negatives += 1

    # 计算Precision、Recall和F1 Score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score
    
    
def cal_acc_batch(model, data_file):
    
    dataset = np.load(data_file)

    x = dataset[:, :-1]  # 数据部分
    y = dataset[:,-1]   # 标签部分
    # print(sample.shape,x,y)

    data = torch.tensor(x, dtype=torch.float32)
    label = torch.tensor(y, dtype=torch.float32)
    #print(dataset.shape, data.shape, label.shape)
    losses = []
    labels_pred = []
    correct = 0

    model = model.eval()
    with torch.no_grad():
        for i in range(len(label)):
            output = model(data[i].unsqueeze(0).cuda())
            prob, pred = output.max(1, keepdim=True)
            loss = torch.nn.CrossEntropyLoss(reduction='sum')(output, label[i].unsqueeze(0).long().cuda()).cpu()
            losses.append(loss.numpy().tolist())
            pred = pred.view(1,).cpu()
            labels_pred.append(pred.numpy()[0])
        
            if pred.eq(label[i].view_as(pred)):
                correct += 1
    acc = correct / len(dataset)
    return acc, [y, labels_pred], losses
