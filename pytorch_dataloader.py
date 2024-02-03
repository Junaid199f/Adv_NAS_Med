#This file implements the dataloaders for different medical datasets
#Written by Muhammad Junaid Ali for NAS-GA Framework

import os
import random

import torch
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.image as mpimg
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import cv2
from PIL import Image
from PIL.Image import Resampling
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset



class MHIST(Dataset):
    def __init__(self,images_path,annotation_file,transforms=None):
        #print(images_path)
        self.annotation_file = pd.read_csv(annotation_file)
        images_names = self.annotation_file['Image Name'].values
        labels = self.annotation_file['Majority Vote Label']
        partitions = self.annotation_file['Partition']
        self.data = []
        for image_name,label,partition in zip(images_names,labels,partitions):
            self.data.append([os.path.join(images_path,image_name),label])
        self.class_map = {"HP": 0, "SSA": 1}
        self.img_dim = (224,224)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        #class_id = torch.tensor([class_id])
        return img_tensor.float(), class_id
        #return img, class_id


class GasHisSDB(Dataset):
    def __init__(self, data_path,transforms=None):
        classes = os.listdir(data_path)
        self.transforms =transforms
        self.data = []
        for class_item in classes :
            for img_path in glob.glob(os.path.join(data_path , class_item) + "/*.png"):
                self.data.append([img_path, class_item])
        self.class_map = {"Abnormal": 0, "Normal": 1}
        self.img_dim = (256, 256)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor.float(), class_id

class kvasir_dataset(Dataset):
    def __init__(self, data_path, transforms=None):
        classes = os.listdir(data_path)
        self.transforms = transforms
        self.data = []
        for class_item in classes:
            for img_path in glob.glob(os.path.join(data_path, class_item) + "/*.jpg"):
                self.data.append([img_path, class_item])
        self.class_map = {"dyed-lifted-polyps": 0, "dyed-resection-margins": 1, "esophagitis": 2, "normal-cecum": 3, "normal-pylorus": 4, "normal-z-line": 5,
                          "polyps": 6, "ulcerative-colitis": 7}
        self.img_dim = (256, 256)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor.float(), class_id
class BreastDataset(Dataset):
    def __init__(self, data_path,transforms=None):
        classes = os.listdir(data_path)
        self.transforms =transforms
        self.data = []
        for class_item in classes :
            for img_path in glob.glob(os.path.join(data_path , class_item) + "/*.png"):
                self.data.append([img_path, class_item])
        self.class_map = {"Malignant Masses": 0, "Benign Masses": 1}
        self.img_dim = (256, 256)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor.float(), class_id
# Define a pytorch dataloader for this dataset
class ocular_toxoplosmosis(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(os.path.join('Datasets','Ocular_Toxoplasmosis_Data_V3','images',self.df['Image name'][index]))
        y = self.df['Label'][index]
        self.class_names = ['healthy', 'active/inactive', 'inactive','active']

        if y== 'healthy':
            y = 0
        elif y == 'active/inactive':
            y=1
        elif y == 'inactive' or  y=='inactive/inactive':
            y=2
        elif y== 'active' or y=='active/active':
            y=3
        else:
            print(y)
            print(1/0)
        class_id = torch.tensor(y)
        img = X.convert('RGB')
        new_width = 256
        new_height = 256
        img = img.resize((new_width, new_height), Resampling.LANCZOS)
        img_tensor = transforms.ToTensor()(img)

        return img_tensor.float(), class_id

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
        img = X.convert('RGB')
        new_width = 256
        new_height = 256
        img = img.resize((new_width, new_height), Resampling.LANCZOS)
        img_tensor = transforms.ToTensor()(img)

        return img_tensor.float(), y

class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x[-3:].lower().endswith('png')]
            print(f'Found {len(images)} {class_name} examples')
            return images

        self.images = {}
        self.class_names = ['normal', 'viral', 'covid']

        for class_name in self.class_names:
            self.images[class_name] = get_images(class_name)

        self.image_dirs = image_dirs
        self.transform = transform

    def __len__(self):
        return sum([len(self.images[class_name]) for class_name in self.class_names])

    def __getitem__(self, index):
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)


class covidr_dataset(Dataset):
    def __init__(self, data_path,transforms=None):
        classes = os.listdir(data_path)
        self.transforms =transforms
        self.data = []
        for class_item in classes :
            for img_path in glob.glob(os.path.join(data_path , class_item) + "/*"):
                self.data.append([img_path, class_item])
        self.class_map = {"Covid": 0, "Normal": 1,'Penumonia':2}
        self.img_dim = (256, 256)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]

        img = Image.open(img_path)
        img = img.convert('RGB')
        new_width = 256
        new_height = 256
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        img_tensor = transforms.ToTensor()(img)
        #img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        #img_tensor = torch.from_numpy(img)
        #img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor.float(), class_id


class covid_dataset(Dataset):
    def __init__(self, data_path,transforms=None):
        classes = os.listdir(data_path)
        self.transforms =transforms
        self.data = []
        for class_item in classes :
            for img_path in glob.glob(os.path.join(data_path , class_item) + "/*"):
                self.data.append([img_path, class_item])
        self.class_map = {"COVID": 0, "Non-COVID": 1}
        self.img_dim = (256, 256)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]

        img = Image.open(img_path)
        img = img.convert('RGB')
        new_width = 256
        new_height = 256
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        img_tensor = transforms.ToTensor()(img)
        #img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        #img_tensor = torch.from_numpy(img)
        #img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor.float(), class_id

class PCBDataset(Dataset):
    def __init__(self, data_path,transforms=None):
        classes = os.listdir(data_path)
        self.transforms =transforms
        self.data = []
        for class_item in classes :
            for img_path in glob.glob(os.path.join(data_path , class_item) + "/*.jpg"):
                self.data.append([img_path, class_item])
        self.class_map = {"basophil": 0, "eosinophil": 1, "erythroblast": 2, "ig": 3, "lymphocyte": 4, "monocyte": 5, "neutrophil":6, "platelet": 7}
        self.img_dim = (256, 256)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor.float(), class_id

