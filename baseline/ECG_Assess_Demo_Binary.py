"""
本代码为心电质量评估二分类示例代码，主要思想是基于经典的SQI-SVM算法实现的，即：先计算心电的SQI，然后利用SVM进行二分类。

"""
import numpy as np
import warnings
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ecg_qc.ecg_qc import EcgQc  #https://github.com/Aura-healthcare/ecg_qc
import glob

####################### Load Data and Label ##########################

dest_dir = 'your dir'

file_data = sorted(glob.glob("{}/*/*_data.npy".format(dest_dir)))
file_label = sorted(glob.glob("{}/*/*_label.npy".format(dest_dir)))

data = []
label = []

for i in file_data:   # load data
    ecg_12L = np.load(i)
    ecg_12L = np.squeeze(ecg_12L)
    for j in range(len(ecg_12L)):
        data.append(ecg_12L[j])
data = np.array(data)

for i in file_label:   # load label
    label_12L = np.load(i)
    for j in label_12L[0]:
        if j == 'A':
            label.append(0)
        else:
            label.append(1)
label = np.array(label)

print(data.shape)
print(label.shape)

####################### SQI Calculate ######################

sqi=[]

for item in range(data.shape[0]):
    ecg_list = data[item].tolist()
    warnings.filterwarnings("ignore")
    ecg_qc = EcgQc('rfc_norm_2s.pkl', sampling_frequency=100, normalized=True)
    sqi_scores = np.array(ecg_qc.compute_sqi_scores(ecg_list))
    sqi.append(sqi_scores[0])
sqi = np.array(sqi)

####################### Check Illegal ######################
nan_mask = np.isnan(sqi)
rows, cols = np.where(nan_mask)
rows = np.unique(rows)
sqi = np.delete(sqi, rows, axis=0)
label = np.delete(label, rows, axis=0) # delete illegal
print(sqi.shape)
print(label.shape)


####################### Train and Test Split ######################

X_train, X_test, y_train, y_test = train_test_split(sqi, label, test_size=0.2, random_state=42)


####################### Train and Test Model ######################

model = svm.SVC(kernel='linear') # init.
model.fit(X_train, y_train)  # train
y_pred = model.predict(X_test) # test

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1_score = f1_score(y_test, y_pred, average='binary')

####################### Performace Metric ######################

print('Acc', accuracy)
print('Pre', precision)
print('Rec', recall)
print('F1', f1_score)

"""
output

(1200, 1000)
(1200,)
(1191, 6)
(1191,)
Acc 0.799163179916318
Pre 0.8923076923076924
Rec 0.7733333333333333
F1 0.8285714285714286
"""




