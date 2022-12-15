#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import sys, os
import random
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# sys.path.append(os.path.abspath(__file__))
PROJ_PATH = "./"
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def idx2label(idx):
    if idx<5:
        return idx+1
    return idx+2

def label2idx(label):
    if label<=5:
        return label-1
    return 5

class Model(nn.Module):
    """
    Two Layer NN
    """
    def __init__(self, input_dim=36, hidden_dim=72, output_dim=6):
        super(Model, self).__init__()
        self.line1 = nn.Linear(input_dim, hidden_dim)
        self.line2 = nn.Linear(hidden_dim, output_dim)
        self.activate = nn.Softmax(dim=-1)

    def forward(self, x):
        h1 = self.line1(x)
        a1 = torch.relu(h1)
        return self.activate(self.line2(a1))
    
class Logistic(nn.Module):
    """
    Logistic Regression
    """
    def __init__(self, input_dim=36, output_dim=6):
        super(Logistic, self).__init__()
        self.line = nn.Linear(input_dim, output_dim)
        self.activate = nn.Softmax(dim=-1)

    def forward(self, x):
        h1 = self.line(x)
        return self.activate(h1)

class Dataset(data.Dataset):
    def __init__(self, df):
        self.x = np.array(df.iloc[:, :-1]) / 255
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(df.iloc[:, -1].map(label2idx), dtype=torch.int64)

    def __getitem__(self, index):     
        return self.x[index, :], self.y[index]

    def __len__(self):
        return int(self.x.shape[0])

def get_data(batch_size=32):
    train_data = pd.read_csv(PROJ_PATH+"/sat.trn", sep=' ', header=None)
    test_data = pd.read_csv(PROJ_PATH+"./sat.tst", sep=' ', header=None)
    train_set = Dataset(train_data)
    test_set = Dataset(test_data)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def EDA():
    train_data = pd.read_csv(PROJ_PATH+"/sat.trn", sep=' ', header=None)
    test_data = pd.read_csv(PROJ_PATH+"./sat.tst", sep=' ', header=None)
    train_label = train_data.iloc[:, -1]
    test_label = test_data.iloc[:, -1]
    
    train_map = {}
    for label in train_label.values:
        if label not in train_map:
            train_map[label] = 1
        else:
            train_map[label] += 1
    sum_ = sum(train_map.values())
    
    plt.pie(np.array(list(train_map.values()))/sum_, labels=train_map.keys(),autopct='%.1f%%')
    plt.show()

    
    test_map = {}
    for label in test_label.values:
        if label not in test_map:
            test_map[label] = 1
        else:
            test_map[label] += 1
    sum_ = sum(test_map.values())
    
    plt.pie(np.array(list(test_map.values()))/sum_, labels=test_map.keys(),autopct='%.1f%%')
    plt.show()

    plt.hist(train_data.iloc[:, 0].values)
    plt.title("Histogram of Pixel Values on the 1st dimension")
    plt.show()

    plt.hist(train_data.iloc[:, 9].values)
    plt.title("Histogram of Pixel Values on the 10th dimension")
    plt.show()

    plt.hist(train_data.iloc[:, 18].values)
    plt.title("Histogram of Pixel Values on the 19th dimension")
    plt.show()

    plt.hist(train_data.iloc[:, 27].values)
    plt.title("Histogram of Pixel Values on the 28th dimension")
    plt.show()

    corrdata=train_data.iloc[:, [0, 9, 18, 27]]
    corr = corrdata.corr()
    plt.figure(figsize=(15,8))
    sns.heatmap(corr, annot=True, annot_kws={"size": 15})
    plt.show()



def train_or_eval(model, epochs=15, batch_size=32, lr=5e-3, l2=1e-2):
    from sklearn.metrics import classification_report, confusion_matrix

    loss_func = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader, test_loader = get_data(batch_size)
    losses = []
    for e in range(epochs):
        total_loss = 0
        y_preds = []
        y_truth = []
        model.train()
        for x, y in train_loader:
            optim.zero_grad()
            y_score = model(x)
            y_pred = y_score.argmax(-1)
            y_pred = y_pred
            y_preds.extend(y_pred.numpy().tolist())
            y_truth.extend(y.numpy().tolist())
            loss = loss_func(y_score, y)
            total_loss += loss.item() * y_pred.shape[0]
            loss.backward()
            optim.step()
        print("training details for epoch ", e+1)
        losses.append(total_loss / len(y_preds))
        print("loss: ", losses[-1])
        print(classification_report(y_truth, y_preds, target_names=['1','2','3','4','5','7']))
        print(confusion_matrix(y_truth, y_preds))
        
        model.eval()
        y_preds = []
        y_truth = []
        with torch.no_grad():
            for x, y in test_loader:
                y_score = model(x)
                y_pred = y_score.argmax(-1)
                y_pred = y_pred
                y_preds.extend(y_pred.numpy().tolist())
                y_truth.extend(y.numpy().tolist()) 
        print("testing details for epoch ", e+1)
        print(classification_report(y_truth, y_preds, target_names=['1','2','3','4','5','7']))
        print(confusion_matrix(y_truth, y_preds))
    return losses     

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix




#%%
from StandardScaler import StandardScaler
train_data = pd.read_csv("sat.trn", sep=' ', header=None)
test_data = pd.read_csv("sat.tst", sep=' ', header=None)
sc = StandardScaler()
x_train = sc.fit_transform(np.array(train_data.iloc[:, :-1]) )
y_train = np.array(list(map(label2idx, train_data.iloc[:, -1])))
x_test = sc.transform(np.array(test_data.iloc[:, :-1]) )
y_test = np.array(list(map(label2idx, test_data.iloc[:, -1])))

#%% Decision tree
from DecisionTree import DecisionTreeClassifier
model_Tree = DecisionTreeClassifier(min_samples_split=2, max_depth=4)
model_Tree.fit(x_train, y_train.reshape([-1,1]))
model_Tree.print_tree()
Y_Pred_Tree = model_Tree.predict(x_train)
print(
    f"Decision Tree Classifier accruacy for training is {accuracy_score(y_train, Y_Pred_Tree)}")
print("Confusion matrix for training:")
print(confusion_matrix(y_train, Y_Pred_Tree))


pred_Test_Tree = model_Tree.predict(x_test)
print(
    f"Test accuracy score of dicision tree is: {accuracy_score(pred_Test_Tree, y_test)}")
print(confusion_matrix(y_test, pred_Test_Tree))


#%%
seed_everything(123)
EDA() # do EDA on predictors and labels
lr_model = Logistic()
lr_losses = train_or_eval(lr_model, epochs=30, batch_size=64, lr=1e-2)
plt.plot(lr_losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
nn_model = Model()
nn_losses = train_or_eval(nn_model, epochs=15, batch_size=64)
plt.plot(nn_losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()