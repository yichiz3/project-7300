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


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix



def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
