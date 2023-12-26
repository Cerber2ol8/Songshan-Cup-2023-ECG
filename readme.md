# ss杯 2023 初赛 心电检测赛道 参赛实现


## 项目介绍       

本项目为2023 ss杯生态应用挑战赛初赛 `赛题3 可穿戴12导联动态心电质量评估 ` 同赛道第二的实现。

实现方式为pytorch搭建CNN+LSTM网络以及Se-ResNet。

提供了数据处理和训练过程的notebook文件，以及测试和评估脚本。

赛题链接：http://nscc.zzu.edu.cn/ai/#/myMatch?tabType=questionsData&tabTeamType=createTeam&tabForumType=matchForum

### 问题描述    
>本赛题使用的数据集来自真实世界的动态心电数据，包含10名患者的标准12导联心电图共100 条。这些心电信号的采样频率为100 Hz，并被分割为每条10秒。数据集标签根据心电信号P、 QRS、T和U波的完整性分为四类：A类代表所有波形清晰可见，无基线漂移，采集质量高。B类 表示存在1-3个干扰心跳，但不会对诊断产生重大影响。C类表示能识别50%以上的波形，对诊断 有部分影响。D类表示完全无法识别波形，没有可用的心电信号。
>本赛题质量评估任务分为两个子题：
>
>1. 二分类质量评估，一类为高质量心电（标签A），另一类为其他（标签B/C/D）。该任务性能 评估指标需统一展示Precision，Recall和F1 Score，并以F1 Score作为算法评价标准。
>
>2. 四分类质量评估，四类标签分别为A/B/C/D。该任务性能指标以Accuracy作为算法评价标准。
>
>赛题数据下载：心电图赛题数据
>
>赛题BaseLine下载：心电图BaseLine
>
>评分标准: 提交模型在私有的测试集上进行预测
>
>1. 二分类质量评估任务性能 评估指标需统一展示Precision，Recall和F1 Score，并以F1 Score作为算法评价标准
>
>2.四分类质量评估任务性能指标以Accuracy作为算法评价标准
>
>提交结果:提交人工智能模型文件

### 使用到的模型
CNN_LSTM 使用多尺度的1d卷积进行特征提取，然后将提取到的特征送入LSTM进行特征识别，最后使用全连接网络进行分类.

Se-ResNet 在Conv1d和LSTM的基础上添加上了SeBlock和ResBlock，形成了多尺度的残差网络模型，相对CNN_LSTM具有更深的层数，但是由于数据集数量较小，效果反而不佳.

### 模型结果

|网络性能和参数信息        ||             |               |Created by  cy  2023.10.25|     |                     |                 |
|:--------:|:-----------:|:-----------:|:-------------:|:------------:|:---------------:|:-------------------:|:---------------:|
|**Model** |**四分类**    |             |**二分类**      |              | **模型参数信息**  |                     |                 |
|          | BestValAcc  | Precison    |Recall         |   F1 Score   | Params          | bidirectional-LSTM  | File            |
|CNN_LSTM   |  83        |  0.9462     |    0.8381     |   0.8888     | 128 64 60 20    |   No                | CNN_LSTM_128_64_60_20_acc83.pt|
|Se-ResNet  |  80        |   0.9375    |    0.8571     |   0.8955     | blk[1 2 2 1]    |   Yes               | Se-ResNet_1221_acc80.pt|


### 项目相关文件
```
main.py # 提交结果运行的主程序，本程序会根据提供的数据集目录生成如下所示的结果
data.ipynb # 数据集代码和相关处理过程
train.ipynb # 训练代码和相关过程
model.py # 模型定义和实现
infer.py # 推理相关功能实现
utils.py # 工具函数
data_raw # 用于存放官方提供数据集的目录
data # 整合的中间数据目录，在data.ipynb中生成
dataset # 训练过程中用用到的数据集，在data.ipynb中生成
CNN_LSTM_128_64_60_20_acc83.pt # 模型权重文件
Se-ResNet_1221_acc80.pt # 模型权重文件
best-ckpt.pt # 模型训练权重检查点文件
baseline # 赛题的baseline实现
```

### 正确输出示例
```
[msxxxxxx@64bb532bde1a heart]$ python main.py
Processing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1200/1200 [00:11<00:00, 108.77iteration/s]
model:      CNN_LSTM  
model:       CNN_LSTM 
Accuracy(classes=4):     0.8941666666666667
Precision(classes=2):    0.9781976744186046
Recall:                  0.9295580110497238
F1 Score:                0.953257790368272

Processing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1200/1200 [00:19<00:00, 62.26iteration/s]
model:       Se-ResNet 
Accuracy(classes=4):     0.9033333333333333
Precision(classes=2):    0.9797687861271677
Recall:                  0.93646408839779
F1 Score:                0.9576271186440679
```
测试用的数据使用的是原始(包含训练）数据集，只是用来展示程序正常输出，不具备参考意义，实际精度数据以测试数据集为准


该项目使用的是zzcs AC平台预置的jupyter环境和conda环境
具体为 jupyterlab-pytorch:1.10.0-centos7.6-dtk-22.10-py37(Jupyter预置) + pytorch1.10.0a0-py37-dtk22.04.2(conda预置)

建议新建Notebook容器，并且使用 jupyterlab-pytorch:1.10.0-centos7.6-dtk-22.10-py37(Jupyter预置)环境，勾选使用加速器
以免出现无所需依赖的问题

该环境中的终端中运行  python main.py 即可正确输出结果