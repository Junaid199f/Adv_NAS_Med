

import numpy as np
import pandas as pd
from PIL.Image import Resampling
from tqdm import tqdm
from glob import glob
from PIL import Image

# pytorch libraries
import torch
from torch import optim,nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms


# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# to make the results are reproducible
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)

data_dir = './Datasets/HAM10000/'
input_size = 224
df_train = pd.read_csv('./Datasets/HAM10000/train.csv')
df_val = pd.read_csv('./Datasets/HAM10000/test.csv')


# Define a pytorch dataloader for this dataset
class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y

#print(df_val)
#print(df_train)
X = Image.open(df_train['path'][2500])
y = torch.tensor(int(df_train['cell_type_idx'][2500]))

print(X)
print(y)

img = X.convert('RGB')
new_width = 256
new_height = 256
img = img.resize((new_width, new_height), Resampling.LANCZOS)
img_tensor = transforms.ToTensor()(img)

print(img_tensor)
print(y)