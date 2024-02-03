import os.path

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

from dataset import Dataset
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
import utils
import json
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
class Evaluate:
    def __init__(self, batch_size):
        self.dataset = Dataset()
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        #self.train_loader, self.valid_loader, self.test_loader, self.classes = self.dataset.inbreast_dataset(self.batch_size)
        #self.train_loader, self.valid_loader, self.test_loader, self.classes = self.dataset.breast_dataset_mias(
             #self.batch_size)
        #self.train_loader, self.valid_loader, self.test_loader, self.classes = self.dataset.breast_dataset_ddsm(
             #self.batch_size)
        #self.train_loader, self.valid_loader, self.test_loader, self.classes = self.dataset.combined_breast_datasets(
             #self.batch_size)
        #self.train_loader, self.valid_loader, self.test_loader, self.classes = self.dataset.kvasir_dataset(self.batch_size)
        self.train_loader,  self.test_loader, self.classes = self.dataset.covid_radiographic_dataset(self.batch_size)
        self.lr = 0.001
        self.scheduler = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def train(self, model, epochs,hash_indv, warmup=False):
        torch.cuda.empty_cache()
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()

        #optimizer = optim.Adam(model.parameters(), lr=0.0001) #0.0001 for breast cancner
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

        # Define the learning rate scheduler
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1) #This is for breast and medmnist dataset
        #scheduler = CosineAnnealingLR(optimizer,T_max = 32, # Maximum number of iterations.
                             #eta_min = 1e-4)  # This is for breast and medmnist dataset
        print('Starting training..')
        for e in range(0, epochs):
            print('=' * 20)
            print(f'Starting epoch {e + 1}/{epochs}')
            print('=' * 20)

            train_loss = 0.
            val_loss = 0.
            model = model.to(device)
            model.train()  # set model to training phase

            for train_step, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs,x = model(images)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()
                train_loss += loss.item()

                if train_step % 20 == 0:
                    print('Evaluating at step', train_step)

                    accuracy = 0

                    model.eval()  # set model to eval phase

                    for val_step, (images, labels) in enumerate(self.train_loader):
                        images = images.to(self.device)
                        labels = labels.to(device)
                        outputs,x = model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                        _, preds = torch.max(outputs, 1)
                        accuracy += sum((preds == labels).cpu().numpy())

                    val_loss /= (val_step + 1)
                    accuracy = accuracy / len(self.test_loader)
                    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

                    model.eval()
                    images, labels = next(iter(self.test_loader))
                    outputs,x = model(images)
                    _, preds = torch.max(outputs, 1)

                    model.train()

                    if accuracy >= 0.98:
                        print('Performance condition satisfied, stopping..')
                        return accuracy

            train_loss /= (train_step + 1)

            print(f'Training Loss: {train_loss:.4f}')
        print('Training complete..')

        # Calculate validation accuracy
        with torch.no_grad():
            val_correct = 0
            val_total = 0
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = torch.squeeze(labels, 1).long().to(device)
                outputs, x = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            val_acc = 100 * val_correct / val_total

        return 100 - val_acc

        # Define the training loop
        # for epoch in range(epochs):
        #     torch.cuda.empty_cache()
        #     running_loss = 0.0
        #     correct = 0
        #     total = 0
        #     for i, (inputs, labels) in enumerate(self.train_loader):
        #         inputs = inputs.to(self.device)
        #         #labels = labels.to(self.device)
        #         labels = torch.squeeze(labels, 1).long().to(device)
        #         model = model.to(self.device)
        #         optimizer.zero_grad()
        #         outputs,x = model(inputs)
        #         loss = criterion(outputs, labels)
        #         nn.utils.clip_grad_norm_(model.parameters(), 5)
        #         loss.backward()
        #         optimizer.step()
        #
        #         running_loss += loss.item()
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()
        #
        #     # Update the learning rate
        #     scheduler.step()
        #
        #     # Calculate validation accuracy
        #     with torch.no_grad():
        #         val_correct = 0
        #         val_total = 0
        #         for inputs, labels in self.valid_loader:
        #             inputs = inputs.to(self.device)
        #             labels = torch.squeeze(labels, 1).long().to(device)
        #             outputs,x = model(inputs)
        #             _, predicted = torch.max(outputs.data, 1)
        #             val_total += labels.size(0)
        #             val_correct += (predicted == labels).sum().item()
        #         val_acc = 100 * val_correct / val_total
        #
        #     print('Epoch [%d], Loss: %.4f, Training Accuracy: %.2f%%, Validation Accuracy: %.2f%%' %
        #           (epoch + 1, running_loss / len(self.train_loader), 100 * correct / total, val_acc))


