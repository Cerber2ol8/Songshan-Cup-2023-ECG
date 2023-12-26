"""
本代码为心电质量评估多分类示例代码，主要思想是使用MLP构建多分类器，直接进行多分类。

"""
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import glob

####################### Load Data and Label ##########################

dest_dir = 'your dir'

file_data = sorted(glob.glob("{}/*/*_data.npy".format(dest_dir)))
file_label = sorted(glob.glob("{}/*/*_label.npy".format(dest_dir)))

data = []
label = []

for i in file_data:  # load data
    ecg_12L = np.load(i)
    ecg_12L = np.squeeze(ecg_12L)
    for j in range(len(ecg_12L)):
        data.append(ecg_12L[j])
data = np.array(data)

for i in file_label:  # load label
    label_12L = np.load(i)
    for j in label_12L[0]:
        if j == 'A':
            label.append(0)
        elif j == 'B':
            label.append(1)
        elif j == 'C':
            label.append(2)
        elif j == 'D':
            label.append(3)
label = np.array(label)

print(data.shape)
print(label.shape)

####################### Train and Test Split ######################

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

####################### Train Model ######################

model = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', random_state=42) #init.
model.fit(X_train, y_train) # train
y_pred = model.predict(X_test) #test
accuracy = accuracy_score(y_test, y_pred)

####################### Performace Metric ######################

print('Acc', accuracy)


"""
output

(1200, 1000)
(1200,)
Acc 0.7833333333333333
"""

