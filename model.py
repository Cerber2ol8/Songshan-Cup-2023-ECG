# Author: LCY
# update: 2023/10/25
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from functools import reduce

class Shrinkage(nn.Module):
    def __init__(self, channel, reduction=4):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.BatchNorm1d(channel // reduction),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y1 = self.gap(x).view(b, c)
        y = self.fc(y1).view(b, c, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 24, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=2, L=32):
        super(SKConv, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()
        for i in range(M):
            self.conv.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 3, stride, padding=1 + i, dilation=1 + i, groups=32, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)))
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Sequential(
            nn.Conv1d(out_channels, d, 1, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU()
        )
        self.fc2 = nn.Conv1d(d, out_channels * M, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        batch_size = input.size(0)
        output = []
        for i, conv in enumerate(self.conv):
            output.append(conv(input))
        U = reduce(lambda x, y: x + y, output)
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.view(batch_size, self.M, -1)
        a_b = list(map(lambda x: x.view(batch_size, self.out_channels, 1), a_b))
        V = list(map(lambda x, y: x * y, output, a_b))
        V = reduce(lambda x, y: x + y, V)
        return V
    
class ResBlk(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(ResBlk, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.se = Shrinkage(out_channel)
        self.sa = SpatialAttention()

        if self.downsample:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channel)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        out = self.se(out) + identity
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.drop1 = nn.Dropout(0)
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.drop2 = nn.Dropout(0)
        self.conv3 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channel * self.expansion)
        self.relu = nn.LeakyReLU()
        self.drop4 = nn.Dropout(0)
        self.downsample = downsample

class ResNet(nn.Module):
    def __init__(self, 
                 block,
                 block_num,
                 cnn_channels=[24,8,8,8],
                 lstm_dims=[16,32,1],
                 num_classes=4, 
                 include_top=True):
        super(ResNet,self).__init__()
        self.include_top = include_top
        self.channels = cnn_channels
        self.lstm_param = lstm_dims
        self.in_channel = cnn_channels[0]
 
        
        #-----------输入网络之前--------------------------
        self.conv1 = nn.Conv1d(1,out_channels=self.in_channel,kernel_size=7,stride=2,padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.channels[0])
        self.relu = nn.ReLU()
        
        "------------------多尺度卷积模块---------------------"
        self.cnn1 = nn.Sequential(
            nn.Conv1d(self.in_channel,  out_channels=self.channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.channels[1]),
            nn.ReLU()
        )
        self.cnn2 = nn.Sequential(
           nn.Conv1d(self.in_channel,  out_channels=self.channels[2], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(self.channels[2]),
            nn.ReLU()
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(self.in_channel, out_channels=self.channels[3], kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(self.channels[3]),
            nn.ReLU()
        )
        
        "-------------LSTM(双向长短时记忆网络)循环卷积网络--------------" 
        self.rnn_layer = nn.LSTM(
            input_size=self.lstm_param[0],
            hidden_size=self.lstm_param[1], 
            num_layers=self.lstm_param[2],
            bidirectional=True,
            dropout=0
        )

        self.se1 = Shrinkage(8)
        self.sa1 = SpatialAttention()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, block_num[0])
        self.layer2 = self._make_layer(block, 64, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 64, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 64, block_num[3], stride=2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.3)
        
        
        if self.include_top:
            self.avgpool = nn.AvgPool1d(1, stride=1)
            self.fc = nn.Linear(64 * 64 * block.expansion, num_classes)
            
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # 使用 Kaiming initialization对Conv1D层初始化
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
    def _make_layer(self,block,channel,block_num, stride=1):          
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channel * block.expansion) #此处用于处理输入特征图
            )
        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = x.view(x.shape[0],1,x.shape[1])
        x1 = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x1)
        output1 = self.cnn1(x)
        output1 = self.se1(output1)
        output2 = self.cnn2(x)
        output2 = self.se1(output2)
        output3 = self.cnn3(x)
        output3 = self.se1(output3)
        output3 = self.dropout(output3)
        x = torch.cat([output1, output2, output3], dim=1)
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.dropout(x)

        if self.include_top:
            x = self.avgpool(x)
            x, _ = self.rnn_layer(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc(x)
        return x

    def resnet18(num_classes=4,include_top=True):
        return ResNet(ResBlk,[2,2,2,2],num_classes=num_classes,include_top=include_top)

    
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        

        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.hardsigmiod = nn.Hardsigmoid()
            
        
    def forward(self, x):
        # Global Average Pooling
        hx = x
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        # Squeeze
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        
        # Excitation
        x = x.view(x.size(0), -1, 1)
        x = x.expand_as(hx)
        
        # Scale the input by the excitation output
        x = x * hx
        
        return x
    
    
class CNN_LSTM(nn.Module):
    def __init__(self, in_dims, hidden_dims, hidden_size, num_classes, dropout=0.0, fc=[60,20], squeeze=False):
        super(CNN_LSTM, self).__init__()
        
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_dims, out_channels=hidden_dims, kernel_size=20, stride=3, padding=10),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims),
            nn.MaxPool1d(kernel_size=2, stride=3)
        )
            
        if squeeze:
            self.conv2 = nn.Sequential(
                nn.Conv1d(in_channels=hidden_dims, out_channels=hidden_dims // 2, kernel_size=7, stride=1, padding=3),
                SEBlock(hidden_dims // 2),
                nn.BatchNorm1d(hidden_dims // 2),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )        
            
            self.conv3 = nn.Sequential(
                nn.Conv1d(in_channels=hidden_dims // 2, out_channels=hidden_dims // 2, kernel_size=10, stride=1, padding=5),
                SEBlock(hidden_dims // 2),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            
        else:
            self.conv2 = nn.Sequential(
                nn.Conv1d(in_channels=hidden_dims, out_channels=hidden_dims // 2, kernel_size=7, stride=1, padding=3),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dims // 2),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )        

            self.conv3 = nn.Sequential(
                nn.Conv1d(in_channels=hidden_dims // 2, out_channels=hidden_dims // 2, kernel_size=10, stride=1, padding=5),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
        
        self.lstm = nn.LSTM(input_size=hidden_dims // 2, hidden_size=hidden_size, num_layers=1, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, fc[0]),
            nn.ReLU(),
            nn.Linear(fc[0], fc[1]),
            nn.ReLU(),
            nn.Linear(fc[1], num_classes),
            nn.Softmax(dim=1)
        )
        

    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Take the output from the last time step
        return x
    
    
def create_model(model_name, *args, **kwargs):
    if model_name == 'se-cnnsltm':
        return CNN_LSTM(*args, **kwargs)
    elif model_name == 'se-resnet':
        return ResNet(ResBlk, *args, **kwargs)
        

    
if __name__ == '__main__':

    input_dim = 1  # 输入信号维度
    hidden_dim = 64
    sequence_length = 1000
    
    #model = EncoderDecoderModel(input_dim, hidden_dim, output_dim, num_layers)
    #model = model = RecurrentAutoencoder(input_size=input_dim, hidden_size=hidden_dim, dropout_ratio=0.0,
                          # n_classes=2, seq_len=sequence_length)
    model = CNN_LSTM(in_dims=input_dim, hidden_dims=128, hidden_size=hidden_dim, num_classes=2, squeeze=True)

    batch_size = 32
    # (sequence_length, batch_size, input_size)
    x = torch.rand(batch_size, sequence_length)
    print(x.shape)
    y  = model(x)
    print(y)
    