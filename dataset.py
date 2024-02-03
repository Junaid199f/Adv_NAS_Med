import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
from PIL import Image
import math
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms, datasets

import utils
from augmentations import Augmentation
import torch
import torch.nn as nn
import torchvision
#This file implements the dataloaders for different medical datasets
#Written by Muhammad Junaid Ali for NAS-GA Framework
import pytorch_dataloader
import os
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
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset
from pytorch_dataloader import MHIST,GasHisSDB
import medmnist
from medmnist import INFO, Evaluator
import torch.utils.data as data

class Dataset:
  def __init__(self):
    self.augmentation = Augmentation()
    self.transforms = self.augmentation.get_augmentation()
  def get_dataset_medmnist(self,dataset_name,batch_size):

      as_rgb =True
      shape_transform = False
      info = INFO[dataset_name]
      task = info['task']
      n_channels = 3 if as_rgb else info['n_channels']
      n_classes = len(info['label'])

      DataClass = getattr(medmnist, info['python_class'])


      print('==> Preparing data...')

      train_transform = utils.Transform3D(mul='random') if shape_transform else utils.Transform3D()
      eval_transform = utils.Transform3D(mul='0.5') if shape_transform else utils.Transform3D()

      train_dataset = DataClass(split='train', transform=train_transform, download='True', as_rgb=as_rgb)
      train_dataset_at_eval = DataClass(split='train', transform=eval_transform, download='True', as_rgb=as_rgb)
      val_dataset = DataClass(split='val', transform=eval_transform, download='True', as_rgb=as_rgb)
      test_dataset = DataClass(split='test', transform=eval_transform, download='True', as_rgb=as_rgb)

      train_loader = data.DataLoader(dataset=train_dataset,
                                     batch_size=batch_size,
                                     shuffle=True)
      train_loader_at_eval = data.DataLoader(dataset=train_dataset_at_eval,
                                             batch_size=batch_size,
                                             shuffle=False)
      val_loader = data.DataLoader(dataset=val_dataset,
                                   batch_size=batch_size,
                                   shuffle=False)
      test_loader = data.DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    shuffle=False)

      return train_loader,val_loader,test_loader
  def get_cellimages(self,batch_size,num_workers = 2):
      # percentage of training set to use as validation
      valid_size = 0.2

      # convert data to a normalized torch.FloatTensor
      # convert data to a normalized torch.FloatTensor
      transform_train = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize((227.9496, 190.6923, 221.5993), (22.5608, 41.5429, 31.3247)),
      ])

      transform_test = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((227.9496, 190.6923, 221.5993), (22.5608, 41.5429, 31.3247)),
      ])

      cellimages_path = os.path.join(os.getcwd(), 'Datasets', 'cell_images')
      # path=r"C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\GasHisSDB\GasHisSDB\160"
      # convert data to a normalized torch.FloatTensor
      dataset = GasHisSDB(cellimages_path, transform_train)
      train_size = int(0.8 * len(dataset))
      test_size = len(dataset) - train_size
      train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
      train_size = int(0.8 * train_size)
      val_size = len(train_dataset) - train_size
      train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
      # dataset = MHIST(r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\images\images',r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\annotations.csv')
      # classes = os.listdir(path) # list of subdirectories and files
      dataloader_train = DataLoader(train_dataset, batch_size=4,
                                    shuffle=True, num_workers=0, drop_last=True)
      dataloader_test = DataLoader(test_dataset, batch_size=4,
                                   shuffle=True, num_workers=0, drop_last=True)
      dataloader_val = DataLoader(valid_dataset, batch_size=4,
                                  shuffle=True, num_workers=0, drop_last=True)
      classes = ['Abnormal', 'Normal']

      return dataloader_train, dataloader_test, dataloader_val, classes
  def ocular_toxoplosmosis(self,batch_size,num_workers = 2):
      mean = (0.49139968, 0.48215827, 0.44653124)
      std = (0.24703233, 0.24348505, 0.26158768)
      train_transform = torchvision.transforms.Compose([
          torchvision.transforms.Resize(size=(300, 300)),
          torchvision.transforms.RandomHorizontalFlip(),
          torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
          torchvision.transforms.RandomHorizontalFlip(),
          torchvision.transforms.RandomRotation(20, interpolation=PIL.Image.BILINEAR),
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(mean=mean, std=std)
      ])

      test_transform = torchvision.transforms.Compose([
          torchvision.transforms.Resize(size=(300, 300)),
          torchvision.transforms.RandomHorizontalFlip(),
          torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
          torchvision.transforms.RandomHorizontalFlip(),
          torchvision.transforms.RandomRotation(20, interpolation=PIL.Image.BILINEAR),
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(mean, std)
      ])
      pbc_dataset_path = os.path.join(os.getcwd(), 'Datasets', 'Ocular_Toxoplasmosis_Data_V3')
      df_train = pd.read_csv('./Datasets/Ocular_Toxoplasmosis_Data_V3/dataset_labels.csv')
      df_val = pd.read_csv('./Datasets/Ocular_Toxoplasmosis_Data_V3/dataset_labels.csv')
      training_set = pytorch_dataloader.ocular_toxoplosmosis(df_train, transform=train_transform)
      validation_set = pytorch_dataloader.ocular_toxoplosmosis(df_val, transform=train_transform)
      train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)
      val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=4)

      return train_loader, val_loader
  def ham10000(self,batch_size,num_workers = 2):
      mean = (0.49139968, 0.48215827, 0.44653124)
      std = (0.24703233, 0.24348505, 0.26158768)
      train_transform = torchvision.transforms.Compose([
          torchvision.transforms.Resize(size=(300, 300)),
          torchvision.transforms.RandomHorizontalFlip(),
          torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
          torchvision.transforms.RandomHorizontalFlip(),
          torchvision.transforms.RandomRotation(20, interpolation=PIL.Image.BILINEAR),
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(mean=mean, std=std)
      ])

      test_transform = torchvision.transforms.Compose([
          torchvision.transforms.Resize(size=(300, 300)),
          torchvision.transforms.RandomHorizontalFlip(),
          torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
          torchvision.transforms.RandomHorizontalFlip(),
          torchvision.transforms.RandomRotation(20, interpolation=PIL.Image.BILINEAR),
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(mean, std)
      ])
      pbc_dataset_path = os.path.join(os.getcwd(), 'Datasets', 'HAM10000')
      df_train = pd.read_csv('./Datasets/HAM10000/train.csv')
      df_val = pd.read_csv('./Datasets/HAM10000/test.csv')
      training_set = pytorch_dataloader.HAM10000(df_train, transform=train_transform)
      validation_set = pytorch_dataloader.HAM10000(df_val, transform=train_transform)
      train_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=4)
      val_loader = DataLoader(validation_set, batch_size=32, shuffle=True, num_workers=4)

      return train_loader, val_loader

  def covid_radiographic_dataset(self,batch_size, num_workers = 2 ):
      train_transform = torchvision.transforms.Compose([
          torchvision.transforms.Resize(size=(300, 300)),
          torchvision.transforms.RandomHorizontalFlip(),
          torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
          torchvision.transforms.RandomHorizontalFlip(),
          torchvision.transforms.RandomRotation(20, interpolation=PIL.Image.BILINEAR),
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])

      test_transform = torchvision.transforms.Compose([
          torchvision.transforms.Resize(size=(300, 300)),
          torchvision.transforms.RandomHorizontalFlip(),
          torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
          torchvision.transforms.RandomHorizontalFlip(),
          torchvision.transforms.RandomRotation(20, interpolation=PIL.Image.BILINEAR),
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
      pbc_dataset_path = os.path.join(os.getcwd(), 'Datasets', 'Covid19R')
      # path=r"C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\GasHisSDB\GasHisSDB\160"
      # convert data to a normalized torch.FloatTensor
      dataset = pytorch_dataloader.covidr_dataset(pbc_dataset_path, test_transform)
      # print(utils.get_mean_and_std(dataset))
      train_size = int(0.8 * len(dataset))
      test_size = len(dataset) - train_size
      train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
      # dataset = MHIST(r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\images\images',r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\annotations.csv')
      # classes = os.listdir(path) # list of subdirectories and files
      dataloader_train = DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=0, drop_last=True)
      dataloader_test = DataLoader(test_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=0, drop_last=True)
      classes = ['Covid', 'Normal', 'Penumonia']
      return  dataloader_train,dataloader_test,classes
  def covid_dataset(self,batch_size,num_workers = 2):
      # percentage of training set to use as validation
      valid_size = 0.2

      # convert data to a normalized torch.FloatTensor
      # convert data to a normalized torch.FloatTensor
      transform_train = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.Resize((224, 224)),
          transforms.Normalize((0.486, 0.486, 0.486), (0.223, 0.223, 0.223)),
          transforms.ToTensor(),
      ])

      transform_test = transforms.Compose([
          transforms.Resize((224, 224)),
          transforms.Normalize((0.486, 0.486, 0.486), (0.223, 0.223, 0.223)),
          transforms.ToTensor(),
      ])

      pbc_dataset_path = os.path.join(os.getcwd(), 'Datasets', '8h65ywd2jr-3','COVID-19 Dataset','COVID-19 Dataset','X-ray')
      # path=r"C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\GasHisSDB\GasHisSDB\160"
      # convert data to a normalized torch.FloatTensor
      dataset = pytorch_dataloader.covid_dataset(pbc_dataset_path, transform_train)
      #print(utils.get_mean_and_std(dataset))
      train_size = int(0.8 * len(dataset))
      test_size = len(dataset) - train_size
      train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
      train_size = int(0.8 * train_size)
      val_size = len(train_dataset) - train_size
      train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
      # dataset = MHIST(r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\images\images',r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\annotations.csv')
      # classes = os.listdir(path) # list of subdirectories and files
      dataloader_train = DataLoader(train_dataset, batch_size=4,
                                    shuffle=True, num_workers=0, drop_last=True)
      dataloader_test = DataLoader(test_dataset, batch_size=4,
                                   shuffle=True, num_workers=0, drop_last=True)
      dataloader_val = DataLoader(valid_dataset, batch_size=4,
                                  shuffle=True, num_workers=0, drop_last=True)
      classes = ['COVID','Non-COVID']

      return dataloader_train, dataloader_test, dataloader_val, classes
  def kvasir_dataset(self,batch_size, num_workers = 2):
      # percentage of training set to use as validation
      valid_size = 0.2

      # convert data to a normalized torch.FloatTensor
      # convert data to a normalized torch.FloatTensor
      transform_train = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize((76.5701, 88.5369, 123.6732), (51.9312, 58.0085, 75.8946)),
      ])

      transform_test = transforms.Compose([
          transforms.ToTensor(),
          transforms.Resize((224, 224)),
          transforms.Normalize((76.5701, 88.5369, 123.6732), (51.9312, 58.0085, 75.8946)),
      ])

      pbc_dataset_path = os.path.join(os.getcwd(), 'Datasets', 'kvasir-dataset','kvasir-dataset')
      # path=r"C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\GasHisSDB\GasHisSDB\160"
      # convert data to a normalized torch.FloatTensor
      dataset = pytorch_dataloader.kvasir_dataset(pbc_dataset_path, transform_train)
      #print(utils.get_mean_and_std(dataset))
      train_size = int(0.8 * len(dataset))
      test_size = len(dataset) - train_size
      train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
      train_size = int(0.8 * train_size)
      val_size = len(train_dataset) - train_size
      train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
      # dataset = MHIST(r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\images\images',r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\annotations.csv')
      # classes = os.listdir(path) # list of subdirectories and files
      dataloader_train = DataLoader(train_dataset, batch_size=4,
                                    shuffle=True, num_workers=0, drop_last=True)
      dataloader_test = DataLoader(test_dataset, batch_size=4,
                                   shuffle=True, num_workers=0, drop_last=True)
      dataloader_val = DataLoader(valid_dataset, batch_size=4,
                                  shuffle=True, num_workers=0, drop_last=True)
      classes = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-cecum', 'normal-pylorus','normal-z-line', 'polyps', 'ulcerative-colitis']

      return dataloader_train, dataloader_test, dataloader_val, classes
  def breast_dataset_mias(self,batch_size,num_workers = 2):
      # percentage of training set to use as validation
      valid_size = 0.2

      # convert data to a normalized torch.FloatTensor
      # convert data to a normalized torch.FloatTensor
      transform_train = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize((58.4890, 58.4890, 58.4890), (62.7576, 62.7576, 62.7576)),
      ])
      transform_test = transforms.Compose([
          transforms.ToTensor(),
          transforms.Resize((224, 224)),
          transforms.Normalize((58.4890, 58.4890, 58.4890), (62.7576, 62.7576, 62.7576)),
      ])

      breast_path = os.path.join(os.getcwd(), 'Datasets', 'MIAS Dataset')
      # path=r"C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\GasHisSDB\GasHisSDB\160"
      # convert data to a normalized torch.FloatTensor
      dataset = pytorch_dataloader.BreastDataset(breast_path, transform_train)
      #print(utils.get_mean_and_std(dataset))
      train_size = int(0.8 * len(dataset))
      test_size = len(dataset) - train_size
      train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
      train_size = int(0.8 * train_size)
      val_size = len(train_dataset) - train_size
      train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
      # dataset = MHIST(r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\images\images',r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\annotations.csv')
      # classes = os.listdir(path) # list of subdirectories and files
      dataloader_train = DataLoader(train_dataset, batch_size=4,
                                    shuffle=True, num_workers=0, drop_last=True)
      dataloader_test = DataLoader(test_dataset, batch_size=4,
                                   shuffle=True, num_workers=0, drop_last=True)
      dataloader_val = DataLoader(valid_dataset, batch_size=4,
                                  shuffle=True, num_workers=0, drop_last=True)
      classes = ['Malignant Masses','Benign Masses']

      return dataloader_train, dataloader_test, dataloader_val, classes
  def inbreast_dataset(self,batch_size,num_workers = 2):
      # percentage of training set to use as validation
      valid_size = 0.2

      # convert data to a normalized torch.FloatTensor
      # convert data to a normalized torch.FloatTensor
      transform_train = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize((37.0704, 37.0704, 37.0704), (41.7564, 41.7564, 41.7564)),
      ])
      transform_test = transforms.Compose([
          transforms.ToTensor(),
          transforms.Resize((224, 224)),
          transforms.Normalize((37.0704, 37.0704, 37.0704), (41.7564, 41.7564, 41.7564)),
      ])

      breast_path = os.path.join(os.getcwd(), 'Datasets', 'INbreast Dataset')
      # path=r"C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\GasHisSDB\GasHisSDB\160"
      # convert data to a normalized torch.FloatTensor
      dataset = pytorch_dataloader.BreastDataset(breast_path, transform_train)
      #print(utils.get_mean_and_std(dataset))
      train_size = int(0.8 * len(dataset))
      test_size = len(dataset) - train_size
      train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
      train_size = int(0.8 * train_size)
      val_size = len(train_dataset) - train_size
      train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
      # dataset = MHIST(r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\images\images',r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\annotations.csv')
      # classes = os.listdir(path) # list of subdirectories and files
      dataloader_train = DataLoader(train_dataset, batch_size=4,
                                    shuffle=True, num_workers=0, drop_last=True)
      dataloader_test = DataLoader(test_dataset, batch_size=4,
                                   shuffle=True, num_workers=0, drop_last=True)
      dataloader_val = DataLoader(valid_dataset, batch_size=4,
                                  shuffle=True, num_workers=0, drop_last=True)
      classes = ['Malignant Masses','Benign Masses']

      return dataloader_train, dataloader_test, dataloader_val, classes
  def combined_breast_datasets(self,batch_size,num_workers = 2):
      # percentage of training set to use as validation
      valid_size = 0.2

      # convert data to a normalized torch.FloatTensor
      # convert data to a normalized torch.FloatTensor
      transform_train = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize((56.7489, 56.7489, 56.7489), (52.4646, 52.4646, 52.4646)),
      ])
      transform_test = transforms.Compose([
          transforms.ToTensor(),
          transforms.Resize((224, 224)),
          transforms.Normalize((56.7489, 56.7489, 56.7489), (52.4646, 52.4646, 52.4646)),
      ])

      breast_path = os.path.join(os.getcwd(), 'Datasets', 'INbreast+MIAS+DDSM Dataset')
      # path=r"C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\GasHisSDB\GasHisSDB\160"
      # convert data to a normalized torch.FloatTensor
      dataset = pytorch_dataloader.BreastDataset(breast_path, transform_train)
      #print(utils.get_mean_and_std(dataset))
      train_size = int(0.8 * len(dataset))
      test_size = len(dataset) - train_size
      train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
      train_size = int(0.8 * train_size)
      val_size = len(train_dataset) - train_size
      train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
      # dataset = MHIST(r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\images\images',r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\annotations.csv')
      # classes = os.listdir(path) # list of subdirectories and files
      dataloader_train = DataLoader(train_dataset, batch_size=4,
                                    shuffle=True, num_workers=0, drop_last=True)
      dataloader_test = DataLoader(test_dataset, batch_size=4,
                                   shuffle=True, num_workers=0, drop_last=True)
      dataloader_val = DataLoader(valid_dataset, batch_size=4,
                                  shuffle=True, num_workers=0, drop_last=True)
      classes = ['Malignant Masses','Benign Masses']

      return dataloader_train, dataloader_test, dataloader_val, classes
  def breast_dataset_ddsm(self,batch_size,num_workers = 2):
      # percentage of training set to use as validation
      valid_size = 0.2

      # convert data to a normalized torch.FloatTensor
      # convert data to a normalized torch.FloatTensor
      transform_train = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize((67.6833, 67.6833, 67.6833), (55.6977, 55.6977, 55.6977)),
      ])
      transform_test = transforms.Compose([
          transforms.ToTensor(),
          transforms.Resize((224, 224)),
          transforms.Normalize((67.6833, 67.6833, 67.6833), (55.6977, 55.6977, 55.6977)),
      ])

      breast_path = os.path.join(os.getcwd(), 'Datasets', 'DDSM Dataset')
      # path=r"C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\GasHisSDB\GasHisSDB\160"
      # convert data to a normalized torch.FloatTensor
      dataset = pytorch_dataloader.BreastDataset(breast_path, transform_train)
      #print(utils.get_mean_and_std(dataset))
      train_size = int(0.8 * len(dataset))
      test_size = len(dataset) - train_size
      train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
      train_size = int(0.8 * train_size)
      val_size = len(train_dataset) - train_size
      train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
      # dataset = MHIST(r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\images\images',r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\annotations.csv')
      # classes = os.listdir(path) # list of subdirectories and files
      dataloader_train = DataLoader(train_dataset, batch_size=4,
                                    shuffle=True, num_workers=0, drop_last=True)
      dataloader_test = DataLoader(test_dataset, batch_size=4,
                                   shuffle=True, num_workers=0, drop_last=True)
      dataloader_val = DataLoader(valid_dataset, batch_size=4,
                                  shuffle=True, num_workers=0, drop_last=True)
      classes = ['Malignant Masses','Benign Masses']

      return dataloader_train, dataloader_test, dataloader_val, classes
  def pbc_dataset(self,batch_size,num_workers = 2):
      # percentage of training set to use as validation
      valid_size = 0.2

      # convert data to a normalized torch.FloatTensor
      # convert data to a normalized torch.FloatTensor
      transform_train = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize((183.9022, 190.8063, 222.7359), (19.3138, 45.8043, 38.8986)),
      ])

      transform_test = transforms.Compose([
          transforms.ToTensor(),
          transforms.Resize((224, 224)),
          transforms.Normalize((183.9022, 190.8063, 222.7359), (19.3138, 45.8043, 38.8986)),
      ])

      pbc_dataset_path = os.path.join(os.getcwd(), 'Datasets', 'PBC_dataset_normal_DIB')
      # path=r"C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\GasHisSDB\GasHisSDB\160"
      # convert data to a normalized torch.FloatTensor
      dataset = pytorch_dataloader.PCBDataset(pbc_dataset_path, transform_train)
      #print(utils.get_mean_and_std(dataset))
      train_size = int(0.8 * len(dataset))
      test_size = len(dataset) - train_size
      train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
      train_size = int(0.8 * train_size)
      val_size = len(train_dataset) - train_size
      train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
      # dataset = MHIST(r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\images\images',r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\annotations.csv')
      # classes = os.listdir(path) # list of subdirectories and files
      dataloader_train = DataLoader(train_dataset, batch_size=4,
                                    shuffle=True, num_workers=0, drop_last=True)
      dataloader_test = DataLoader(test_dataset, batch_size=4,
                                   shuffle=True, num_workers=0, drop_last=True)
      dataloader_val = DataLoader(valid_dataset, batch_size=4,
                                  shuffle=True, num_workers=0, drop_last=True)
      classes = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

      return dataloader_train, dataloader_test, dataloader_val, classes
  def get_mhist(self,batch_size,num_workers = 2):
      MHIST_path = os.path.join(os.getcwd(), 'Datasets','DCPHB','images','images')
      MHIST_annoation_path = os.path.join(os.getcwd(),'Datasets','DCPHB','annotations.csv')
      # percentage of training set to use as validation
      valid_size = 0.2

      # convert data to a normalized torch.FloatTensor
      transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        #transforms.Normalize((198.2438, 166.4309, 188.5556), (41.3462, 58.2809, 47.7798)),
      ])

      transform_test = transforms.Compose([
          transforms.ToTensor()
          #transforms.Normalize((198.2438, 166.4309, 188.5556), (41.3462, 58.2809, 47.7798)),
      ])

      dataset = MHIST(MHIST_path, MHIST_annoation_path , transform_train)
      train_size = int(0.8 * len(dataset))
      test_size = len(dataset) - train_size
      train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
      train_size = int(0.8 * train_size)
      val_size = len(train_dataset) - train_size
      train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
      # dataset = MHIST(r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\images\images',r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\annotations.csv')
      # classes = os.listdir(path) # list of subdirectories and files
      dataloader_train = DataLoader(train_dataset, batch_size=4,
                                    shuffle=True, num_workers=0,drop_last=True)
      dataloader_test = DataLoader(test_dataset, batch_size=4,
                                   shuffle=True, num_workers=0,drop_last=True)
      dataloader_val = DataLoader(valid_dataset, batch_size=4,
                                  shuffle=True, num_workers=0,drop_last=True)
      classes = ['SSA', 'HP']

      return dataloader_train, dataloader_test, dataloader_val, classes
  def get_gashisdb(self,batch_size,num_workers=2):
      # percentage of training set to use as validation
      valid_size = 0.2

      # convert data to a normalized torch.FloatTensor
      # convert data to a normalized torch.FloatTensor
      transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((227.9496, 190.6923, 221.5993), (22.5608, 41.5429, 31.3247)),
      ])

      transform_test = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((227.9496, 190.6923, 221.5993), (22.5608, 41.5429, 31.3247)),
      ])


      GasHisSDB_path = os.path.join(os.getcwd(),'GasHisSDB','GasHisSDB','160')
      # path=r"C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\GasHisSDB\GasHisSDB\160"
      # convert data to a normalized torch.FloatTensor
      dataset = GasHisSDB(GasHisSDB_path, transform_train)
      train_size = int(0.8 * len(dataset))
      test_size = len(dataset) - train_size
      train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
      train_size = int(0.8 * train_size)
      val_size = len(train_dataset) - train_size
      train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
      # dataset = MHIST(r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\images\images',r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\annotations.csv')
      # classes = os.listdir(path) # list of subdirectories and files
      dataloader_train = DataLoader(train_dataset, batch_size=4,
                                    shuffle=True, num_workers=0,drop_last=True)
      dataloader_test = DataLoader(test_dataset, batch_size=4,
                                   shuffle=True, num_workers=0,drop_last=True)
      dataloader_val = DataLoader(valid_dataset, batch_size=4,
                                  shuffle=True, num_workers=0,drop_last=True)
      classes = ['Abnormal','Normal']


      return dataloader_train,dataloader_test,dataloader_val,classes
  def breast_dataset_ddsm(self,batch_size,num_workers = 2):
      # percentage of training set to use as validation
      valid_size = 0.2

      # convert data to a normalized torch.FloatTensor
      # convert data to a normalized torch.FloatTensor
      transform_train = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize((67.6833, 67.6833, 67.6833), (55.6977, 55.6977, 55.6977)),
      ])
      transform_test = transforms.Compose([
          transforms.ToTensor(),
          transforms.Resize((224, 224)),
          transforms.Normalize((67.6833, 67.6833, 67.6833), (55.6977, 55.6977, 55.6977)),
      ])

      breast_path = os.path.join(os.getcwd(), 'Datasets', 'DDSM Dataset')
      # path=r"C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\GasHisSDB\GasHisSDB\160"
      # convert data to a normalized torch.FloatTensor
      dataset = pytorch_dataloader.BreastDataset(breast_path, transform_train)
      #print(utils.get_mean_and_std(dataset))
      train_size = int(0.8 * len(dataset))
      test_size = len(dataset) - train_size
      train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
      train_size = int(0.8 * train_size)
      val_size = len(train_dataset) - train_size
      train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
      # dataset = MHIST(r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\images\images',r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\annotations.csv')
      # classes = os.listdir(path) # list of subdirectories and files
      dataloader_train = DataLoader(train_dataset, batch_size=4,
                                    shuffle=True, num_workers=0, drop_last=True)
      dataloader_test = DataLoader(test_dataset, batch_size=4,
                                   shuffle=True, num_workers=0, drop_last=True)
      dataloader_val = DataLoader(valid_dataset, batch_size=4,
                                  shuffle=True, num_workers=0, drop_last=True)
      classes = ['Malignant Masses','Benign Masses']

      return dataloader_train, dataloader_test, dataloader_val, classes
  def get_dataset_cifar100(self,batch_size,num_workers=2):
    # percentage of training set to use as validation
    valid_size = 0.2

    # convert data to a normalized torch.FloatTensor
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    #train_transform, valid_transform = eeea_transforms._data_transforms_cifar10_search()
    # choose the training and test datasets
    train_data = datasets.CIFAR100('data', train=True,
                                  download=True, transform=transform_train)
    test_data = datasets.CIFAR100('data', train=False,
                                download=True, transform=transform_test)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
        num_workers=num_workers)

    # specify the image classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']
    return train_loader,valid_loader,test_loader,classes
  def get_dataset_cifar10(self,batch_size,num_workers=2):
    # percentage of training set to use as validation
    valid_size = 0.2

    # convert data to a normalized torch.FloatTensor
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    #train_transform, valid_transform = eeea_transforms._data_transforms_cifar10_search()
    # choose the training and test datasets
    train_data = datasets.CIFAR10('data', train=True,
                                  download=True, transform=transform_train)
    test_data = datasets.CIFAR10('data', train=False,
                                download=True, transform=transform_test)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
        num_workers=num_workers)

    # specify the image classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']
    return train_loader,valid_loader,test_loader,classes
