import random
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary

from ga import GA
from mo_ga import MOGA,NAS

if __name__ == '__main__':
  #ga = GA(20,20,0.9,0.6,32,2,3,20,2,2,16,0.3,True,True,True)
  #ga.evolve()
  #Multi-Objective
  #moga = MOGA(20,20,0.9,0.6,32,2,3,20,2,2,16,0.3,True,True,True)
  #MOGA.evolve()
  #Single Objective
  # soga = MOGA(20, 20, 0.9, 0.6, 32, 2, 3, 20, 2, 2, 16, 0.3, True, True, True)
  # soga.