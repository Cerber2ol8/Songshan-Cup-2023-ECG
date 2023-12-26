# Author: LCY
# update: 2023/10/25
import os
import torch
import argparse
import numpy as np
from infer import load_model, load_data, cal_acc

########################################################################################################################################################
#                                           网络性能和参数信息                                  Created by Lcy  2023.10.25
########################################################################################################################################################
#        四分类  Val      |              二分类   Val              |                          模型参数信息 
########################################################################################################################################################
#  Model     BestValAcc  |    Precison     Recall      F1 Score   |         Params              bidirectional-LSTM             File
# CNN_LSTM      83       |    0.9462       0.8381      0.8888     |       128 64 60 20              No                   CNN_LSTM_128_64_60_20_acc83.pt
# Se-ResNet     80       |    0.9375       0.8571      0.8955     |       blk[1 2 2 1]              Yes                  Se-ResNet_1221_acc80.pt
########################################################################################################################################################


########################################################################################################################################################
#                                                项目介绍       
########################################################################################################################################################
# 使用到的模型
# CNN_LSTM 使用多尺度的1d卷积进行特征提取，然后将提取到的特征送入LSTM进行特征识别，最后使用全连接网络进行分类.

# Se-ResNet 在Conv1d和LSTM的基础上添加上了SeBlock和ResBlock，形成了多尺度的残差网络模型，相对CNN_LSTM具有更深的层数，但是由于数据集数量较小，效果反而不佳.

# 相关文件
# main.py 提交结果运行的主程序，本程序会根据提供的数据集目录生成如下所示的结果
# data.ipynb 数据集代码和相关处理过程
# train.ipynb 训练代码和相关过程
# model.py 模型定义和实现
# infer.py 推理相关功能实现
# utils.py 工具函数
# data_raw 用于存放官方提供数据集的目录
# data 整合的中间数据目录，在data.ipynb中生成
# dataset 训练过程中用用到的数据集，在data.ipynb中生成
# CNN_LSTM_128_64_60_20_acc83.pt 模型权重文件
# Se-ResNet_1221_acc80.pt 模型权重文件
# best-ckpt.pt 模型训练权重检查点文件


# 程序正确输出示例

# [msliuzy@64bb532bde1a heart]$ python main.py
# Processing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1200/1200 [00:11<00:00, 108.77iteration/s]
# model:      CNN_LSTM  
# model:       CNN_LSTM 
# Accuracy(classes=4):     0.8941666666666667
# Precision(classes=2):    0.9781976744186046
# Recall:                  0.9295580110497238
# F1 Score:                0.953257790368272

# Processing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1200/1200 [00:19<00:00, 62.26iteration/s]
# model:       Se-ResNet 
# Accuracy(classes=4):     0.9033333333333333
# Precision(classes=2):    0.9797687861271677
# Recall:                  0.93646408839779
# F1 Score:                0.9576271186440679

# 测试用的数据使用的是原始(包含训练）数据集，只是用来展示程序正常输出，不具备参考意义，实际精度数据以测试情况为准


# 该项目使用的是zzcs AC平台预置的jupyter环境和conda环境
# 具体为 jupyterlab-pytorch:1.10.0-centos7.6-dtk-22.10-py37(Jupyter预置) + pytorch1.10.0a0-py37-dtk22.04.2(conda预置)

# 建议新建Notebook容器，并且使用 jupyterlab-pytorch:1.10.0-centos7.6-dtk-22.10-py37(Jupyter预置)环境，勾选使用加速器
# 以免出现无所需依赖的问题

# 该环境中的终端中运行  python main.py 即可正确输出结果



def test(data_dir):
    print('\r\n\r\n\r\n')
    # CNN_LSTM
    model = load_model("CNN_LSTM_128_64_60_20_acc83.pt", model_id=0)
    data, label = load_data(data_dir)
    acc, prec, recall, f1 = cal_acc(model, data, label)
    result('CNN_LSTM', acc, prec, recall, f1)
    # Se-ResNet
    model = load_model("Se-ResNet_1221_acc80.pt", model_id=1)
    data, label = load_data(data_dir)
    acc, prec, recall, f1 = cal_acc(model, data, label)
    result('Se-ResNet', acc, prec, recall, f1)


    
def result(model_name, acc, prec, recall, f1):
    print(f'model:       { model_name } ')
    print(f'(4 classes)')
    print(f'Accuracy(classes=4):    { acc }')
    print(f'(2 classes)')
    print(f'Precision(classes=2):   { prec }')
    print(f'Recall:                 { recall }')
    print(f'F1 Score:               { f1 }')
    print('\r\n\r\n\r\n')
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="data_raw") # 测试时加载数据的位置
    args = parser.parse_args()
    test(args.test_dir)

    
    
if __name__ == '__main__':
    main()