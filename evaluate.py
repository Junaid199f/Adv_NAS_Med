from __future__ import division
import copy
import logging
import os.path
import random
import time
from collections import OrderedDict

import cv2
from matplotlib import pyplot as plt
import PIL
import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch import optim
from torch.backends import cudnn

import torchattacks
import torchvision.transforms as transforms
from medmnist.info import INFO, DEFAULT_ROOT
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
from attacks import fgsm_attack
from dataset import Dataset
from utils import (upper_limit, lower_limit, clamp, attack_pgd, evaluate_pgd, evaluate_standard, evaluate_fgsm)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # <<<<<<<<<<<<<<<<<<<<
from evaluation_measures import evaluate_measures
# Loading adverserial attack libraries


from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod
from art.estimators.classification import PyTorchClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0

class LinfPGDAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y,epsilon,k,criterion,alpha):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits,_ = self.model(x)
                y = y.squeeze()
                y = y.long()
                y = y.view(-1)
                loss = criterion(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

def attack(x, y, model, adversary):
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y)
    return adv

class Evaluate:
    def __init__(self, batch_size, dataset_name, medmnist_dataset):
        self.dataset = Dataset()
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.medmnist_dataset = medmnist_dataset
        self.optimizer = None
        # Checking for the dataset
        if dataset_name == 'CIFAR-10':
            self.train_loader, self.valid_loader, self.test_loader, self.classes = self.dataset.get_dataset(
                self.batch_size)
        elif dataset_name == 'CIFAR-100':
            self.train_loader, self.valid_loader, self.test_loader = self.dataset.get_dataset_cifar100(self.batch_size)
        elif dataset_name == 'FASHIONMNIST':
            self.train_loader, self.valid_loader, self.test_loader = self.dataset.get_dataset_fashionmnist(
                self.batch_size)
        elif dataset_name == 'IMAGENET':
            self.train_loader, self.valid_loader, self.test_loader = self.dataset.get_dataset_imagenet(self.batch_size)
        elif dataset_name == 'MEDMNIST':
            self.train_loader, self.valid_loader, self.test_loader = self.dataset.get_dataset_medmnist(medmnist_dataset,
                                                                                                       self.batch_size)
        elif dataset_name == 'GASHISSDB':
            self.train_loader, self.valid_loader, self.test_loader, classes = self.dataset.get_gashisdb(self.batch_size)
        elif dataset_name == 'MHIST':
            self.train_loader, self.valid_loader, self.test_loader, classes = self.dataset.get_mhist(self.batch_size)
        # self.train_loader, self.valid_loader, self.test_loader, self.classes = self.dataset.get_dataset(self.batch_size)
        self.lr = 0.001
        self.scheduler = None


    def __train(self, model, train_loader, task, criterion, optimizer, device, writer):
        total_loss = []
        global iteration

        model.train()
        # Training the model
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs, x = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)

            total_loss.append(loss.item())
            writer.add_scalar('train_loss_logs', loss.item(), iteration)
            iteration += 1

            loss.backward()
            optimizer.step()

        epoch_loss = sum(total_loss) / len(total_loss)
        return epoch_loss

    def __test(self, model, evaluator, data_loader, task, criterion, device, run, type_task, save_folder=None):
        # Testing the model
        check_evaluator = medmnist.Evaluator(self.medmnist_dataset, type_task)
        info = INFO[self.medmnist_dataset]
        task = info["task"]
        root = DEFAULT_ROOT
        npz_file = np.load(os.path.join(root, "{}.npz".format((self.medmnist_dataset))))
        if type_task == 'train':
            self.labels = npz_file['train_labels']
        elif type_task == 'val':
            self.labels = npz_file['val_labels']
        elif type_task == 'test':
            self.labels = npz_file['test_labels']
        else:
            raise ValueError

        model.eval()

        total_loss = []
        y_score = torch.tensor([]).to(device)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                outputs, x = model(inputs.to(device))

                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32).to(device)
                    loss = criterion(outputs, targets)
                    m = nn.Sigmoid()
                    outputs = m(outputs).to(device)
                else:
                    targets = torch.squeeze(targets, 1).long().to(device)
                    loss = criterion(outputs, targets)
                    m = nn.Softmax(dim=1)
                    outputs = m(outputs).to(device)
                    targets = targets.float().resize_(len(targets), 1)

                total_loss.append(loss.item())
                y_score = torch.cat((y_score, outputs), 0)

            y_score = y_score.detach().cpu().numpy()
            auc, acc = evaluator.evaluate(y_score, save_folder, run)
            f1 = evaluate_measures(self.labels, y_score, task)
            test_loss = sum(total_loss) / len(total_loss)


            return [test_loss, auc, acc, f1]

    def train(self, model, epochs,  grad_clip, evaluation, data_flag, output_root, num_epochs, gpu_ids,
              batch_size, download, run,is_final):
        print(data_flag,"\n")
        as_rgb = True
        resize = False
        info = INFO[data_flag]
        task = info['task']
        channels = info['n_channels']
        n_channels = 3
        n_classes = len(info['label'])

        lr = 0.001
        gamma = 0.1
        milestones = [0.5 * num_epochs, 0.75 * num_epochs]

        info = INFO[data_flag]
        task = info['task']
        n_channels = 3 if as_rgb else info['n_channels']
        n_classes = len(info['label'])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = nn.DataParallel(model)
        model = model.to(device)
        DataClass = getattr(medmnist, info['python_class'])
        # mean = 0.
        # std = 0.
        # for images, _ in self.train_loader:
        #     batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        #     images = images.view(batch_samples, images.size(1), -1)
        #     mean += images.mean(2).sum(0)
        #     std += images.std(2).sum(0)
        #
        # mean /= len(self.train_loader.dataset)
        # std /= len(self.train_loader.dataset)

        # if channels==1:
        #     mu = torch.tensor(mean).view(1, 1, 1).cuda()
        #     std = torch.tensor(std).view(1, 1, 1).cuda()
        # else:
        # mu = torch.tensor(mean).view(3, 1, 1).cuda()
        # std = torch.tensor(std).view(3, 1, 1).cuda()

        str_ids = gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                gpu_ids.append(id)
        if len(gpu_ids) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])

        device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')

        output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
        print('==> Preparing data...')

        if resize:
            data_transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Modify normalization based on your dataset
            ])
        else:
            data_transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomRotation(15),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
                transforms.Resize((32, 32), interpolation=PIL.Image.NEAREST),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Modify normalization based on your dataset
            ])

        train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb)
        val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb)
        test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb)

        train_loader = data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
        train_loader_at_eval = data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
        val_loader = data.DataLoader(dataset=val_dataset,
                                     batch_size=batch_size,
                                     shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)

        print('==> Building and training model...')

        train_evaluator = medmnist.Evaluator(data_flag, 'train')
        val_evaluator = medmnist.Evaluator(data_flag, 'val')
        test_evaluator = medmnist.Evaluator(data_flag, 'test')

        if task == "multi-label, binary-class":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        net = model
        net = net.to(device)
        net = torch.nn.DataParallel(net)
        model = net
        cudnn.benchmark = True

        adversary = LinfPGDAttack(net)
        criterion = criterion
        learning_rate = 0.1
        epsilon = 0.00314
        k = 7
        alpha = 0.000784
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)
        for epoch in range(epochs):
            print('\n[ Train epoch: %d ]' % epoch)
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                k = 7
                adv = adversary.perturb(inputs, targets,epsilon,k,criterion,alpha)
                adv_outputs,_ = model(adv)
                targets = targets.squeeze()
                targets = targets.long()
                targets = targets.view(-1)
                loss = criterion(adv_outputs, targets)
                loss.backward()

                optimizer.step()
                train_loss += loss.item()
                _, predicted = adv_outputs.max(1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # if batch_idx % 10 == 0:
                # print('\nCurrent batch:', str(batch_idx))
                # print('Current adversarial train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                # print('Current adversarial train loss:', loss.item())

            print('\nTotal adversarial train accuarcy:', 100. * correct / total)
            print('Total adversarial train loss:', train_loss)

            print('\n[ Test epoch: %d ]' % epoch)
            model_test = model.eval()
            benign_loss = 0
            adv_loss = 0
            benign_correct = 0
            adv_correct = 0
            total = 0
            with torch.no_grad():
                if is_final == True:
                    for batch_idx, (inputs, targets) in enumerate(test_loader):
                        inputs, targets = inputs.to(device), targets.to(device)
                        total += targets.size(0)

                        outputs,_ = model_test(inputs)
                        targets = targets.squeeze()
                        targets = targets.long()
                        targets = targets.view(-1)
                        loss = criterion(outputs, targets)
                        benign_loss += loss.item()

                        _, predicted = outputs.max(1)
                        benign_correct += predicted.eq(targets).sum().item()

                        # if batch_idx % 10 == 0:
                        # print('\nCurrent batch:', str(batch_idx))
                        # print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                        # print('Current benign test loss:', loss.item())
                        k=7
                        adv = adversary.perturb(inputs, targets,epsilon,k,criterion,alpha)
                        adv_outputs,_ = model_test(adv)
                        loss = criterion(adv_outputs, targets)
                        adv_loss += loss.item()

                        _, predicted = adv_outputs.max(1)
                        adv_correct += predicted.eq(targets).sum().item()

                        if batch_idx % 10 == 0:
                            print('Current adversarial test accuracy:',
                                  str(predicted.eq(targets).sum().item() / targets.size(0)))
                            print('Current adversarial test loss:', loss.item())

                    print('\nTotal benign test accuarcy:', 100. * benign_correct / total)
                    print('Total adversarial test Accuarcy:', 100. * adv_correct / total)
                    print('Total benign test loss:', benign_loss)
                    print('Total adversarial test loss:', adv_loss)
                else:
                    for batch_idx, (inputs, targets) in enumerate(val_loader):
                        inputs, targets = inputs.to(device), targets.to(device)
                        total += targets.size(0)

                        outputs, _ = model_test(inputs)
                        targets = targets.squeeze()
                        targets = targets.long()
                        targets = targets.view(-1)
                        loss = criterion(outputs, targets)
                        benign_loss += loss.item()

                        _, predicted = outputs.max(1)
                        benign_correct += predicted.eq(targets).sum().item()

                        # if batch_idx % 10 == 0:
                        # print('\nCurrent batch:', str(batch_idx))
                        # print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                        # print('Current benign test loss:', loss.item())
                        k = 7
                        adv = adversary.perturb(inputs, targets, epsilon, k, criterion, alpha)
                        adv_outputs, _ = model_test(adv)
                        loss = criterion(adv_outputs, targets)
                        adv_loss += loss.item()

                        _, predicted = adv_outputs.max(1)
                        adv_correct += predicted.eq(targets).sum().item()

                        if batch_idx % 10 == 0:
                            print('Current adversarial test accuracy:',
                                  str(predicted.eq(targets).sum().item() / targets.size(0)))
                            print('Current adversarial test loss:', loss.item())

                    print('\nTotal benign test accuarcy:', 100. * benign_correct / total)
                    print('Total adversarial test Accuarcy:', 100. * adv_correct / total)
                    print('Total benign test loss:', benign_loss)
                    print('Total adversarial test loss:', adv_loss)

        # plt.plot(train_loss_epoch, label='Train Loss')
        # plt.plot(val_loss_epoch, label='Validation Loss')
        # plt.plot(train_acc_epoch, label='Train Accuracy')
        # plt.plot(val_acc_epoch, label='Validation Accuracy')
        # #plt.legend()
        # plt.show()



        # sns.set(style="darkgrid")
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.plot(train_loss_epoch, label='Train Loss')
        # plt.plot(val_loss_epoch, label='Validation Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.subplot(1, 2, 2)
        # plt.plot(train_acc_epoch, label='Train Accuracy')
        # plt.plot(val_acc_epoch, label='Validation Accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        #
        # plt.show()

        # Evaluation
        model_test = model_test.cuda()
        #model_test.load_state_dict(best_state_dict)
        model_test.float()
        model_test.eval()

        # pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, criterion, 50, 10, task, device, std)
        #fgsm_loss, fgsm_acc = evaluate_fgsm(test_loader, model_test, criterion, 7, 10, task, device, std)
        # pgd_loss, pgd_acc = evaluate_fgsm(test_loader, model_test, criterion, 50, 10, task, device, std)
        # test_loss, test_acc = evaluate_standard(test_loader, model_test, criterion, task, device, std)
        #
        # logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        # logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
        # print('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        # print('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
        # #print('Test Loss \t Test Acc \t FGSM Loss \t FGSM Acc')
        #print('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, fgsm_loss, fgsm_acc)
        # alpha = (2 / 255.) / std

        # Check the adverserial attacks on the trained model
        if is_final:
            #Evaluate it on clean dataset
            val_correct = 0
            val_total = 0
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                # for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = torch.squeeze(labels, 1).long().to(device)
                target_labels = (labels + 1) % 10
                # shift all class loops one to the right, 1=>2, 2=>3, .., 9=>0
                for i in range(inputs.shape[0]):
                    min_ele = torch.min(inputs[i])
                    inputs[i] -= min_ele
                    inputs[i] /= torch.max(inputs[i])
                predictions, _ = model_test(inputs)
                # m = nn.Softmax(dim=1)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                _, predicted = torch.max(predictions, 1)
                # predicted = predictions
                # t = torch.from_numpy(predictions)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            val_acc_pgd = 100 * val_correct / val_total
            print(" Accuracy on clean dataset %.2f%%" % val_acc_pgd)
            # PGD Attack
            val_correct = 0
            val_total = 0
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                # for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = torch.squeeze(labels, 1).long().to(device)
                atk = torchattacks.PGD(model_test, epsilon, alpha)
                atk.set_mode_targeted_by_label(quiet=True)  # do not show the message
                target_labels = (labels + 1) % 10
                # shift all class loops one to the right, 1=>2, 2=>3, .., 9=>0
                for i in range(inputs.shape[0]):
                    min_ele = torch.min(inputs[i])
                    inputs[i] -= min_ele
                    inputs[i] /= torch.max(inputs[i])
                attack1 = torchattacks.PGD(model_test, epsilon, alpha)
                adv_images = attack1(inputs, labels)
                predictions, _ = model_test(adv_images)
                # m = nn.Softmax(dim=1)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                _, predicted = torch.max(predictions, 1)
                # predicted = predictions
                # t = torch.from_numpy(predictions)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            val_acc_pgd = 100 * val_correct / val_total
            print(" Accuracy after PGD adverserial attack %.2f%%" % val_acc_pgd)


            #Check with PGD 20
            # PGD Attack
            val_correct = 0
            val_total = 0
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                # for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = torch.squeeze(labels, 1).long().to(device)
                atk = torchattacks.PGD(model_test, epsilon, alpha,20)
                atk.set_mode_targeted_by_label(quiet=True)  # do not show the message
                target_labels = (labels + 1) % 10
                # shift all class loops one to the right, 1=>2, 2=>3, .., 9=>0
                for i in range(inputs.shape[0]):
                    min_ele = torch.min(inputs[i])
                    inputs[i] -= min_ele
                    inputs[i] /= torch.max(inputs[i])
                attack1 = torchattacks.PGD(model_test, epsilon, alpha)
                adv_images = attack1(inputs, labels)
                predictions, _ = model_test(adv_images)
                # m = nn.Softmax(dim=1)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                _, predicted = torch.max(predictions, 1)
                # predicted = predictions
                # t = torch.from_numpy(predictions)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            val_acc_pgd = 100 * val_correct / val_total
            print(" Accuracy after PGD 20 adverserial attack %.2f%%" % val_acc_pgd)
            # Check with PGD 50
            # PGD Attack
            val_correct = 0
            val_total = 0
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                # for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = torch.squeeze(labels, 1).long().to(device)
                atk = torchattacks.PGD(model_test, epsilon, alpha,50)
                atk.set_mode_targeted_by_label(quiet=True)  # do not show the message
                target_labels = (labels + 1) % 10
                # shift all class loops one to the right, 1=>2, 2=>3, .., 9=>0
                for i in range(inputs.shape[0]):
                    min_ele = torch.min(inputs[i])
                    inputs[i] -= min_ele
                    inputs[i] /= torch.max(inputs[i])
                attack1 = torchattacks.PGD(model_test, epsilon, alpha)
                adv_images = attack1(inputs, labels)
                predictions, _ = model_test(adv_images)
                # m = nn.Softmax(dim=1)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                _, predicted = torch.max(predictions, 1)
                # predicted = predictions
                # t = torch.from_numpy(predictions)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            val_acc_pgd = 100 * val_correct / val_total
            print(" Accuracy after PGD 50 adverserial attack %.2f%%" % val_acc_pgd)
            # Check with PGD 100
            # PGD Attack
            val_correct = 0
            val_total = 0
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                # for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = torch.squeeze(labels, 1).long().to(device)
                atk = torchattacks.PGD(model_test, epsilon, alpha)
                atk.set_mode_targeted_by_label(quiet=True)  # do not show the message
                target_labels = (labels + 1) % 10
                # shift all class loops one to the right, 1=>2, 2=>3, .., 9=>0
                for i in range(inputs.shape[0]):
                    min_ele = torch.min(inputs[i])
                    inputs[i] -= min_ele
                    inputs[i] /= torch.max(inputs[i])
                attack1 = torchattacks.PGD(model_test, epsilon, alpha,100)
                adv_images = attack1(inputs, labels)
                predictions, _ = model_test(adv_images)
                # m = nn.Softmax(dim=1)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                _, predicted = torch.max(predictions, 1)
                # predicted = predictions
                # t = torch.from_numpy(predictions)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            val_acc_pgd = 100 * val_correct / val_total
            print(" Accuracy after PGD  100 adverserial attack %.2f%%" % val_acc_pgd)

            # FGSM Attack
            val_correct = 0
            val_total = 0
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                # for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = torch.squeeze(labels, 1).long().to(device)
                atk = torchattacks.FFGSM(model_test, epsilon, alpha)
                atk.set_mode_targeted_by_label(quiet=True)  # do not show the message
                target_labels = (labels + 1) % 10
                # shift all class loops one to the right, 1=>2, 2=>3, .., 9=>0
                for i in range(inputs.shape[0]):
                    min_ele = torch.min(inputs[i])
                    inputs[i] -= min_ele
                    inputs[i] /= torch.max(inputs[i])
                attack1 = torchattacks.FFGSM(model_test, epsilon, alpha)
                adv_images = attack1(inputs, labels)
                predictions,_ = model_test(adv_images)
                #m = nn.Softmax(dim=1)
                #outputs = m(predictions).to(device)
                #_, predicted = torch.max(outputs.data, 1)
                _, predicted = torch.max(predictions, 1)
                # predicted = predictions
                # t = torch.from_numpy(predictions)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            val_acc_fgsm = 100 * val_correct / val_total
            print(" Accuracy after FGSM adverserial attack %.2f%%" % val_acc_fgsm)
            # BIM Attack
            val_correct = 0
            val_total = 0
            for batch_idx, (inputs, labels) in enumerate(test_loader):

                inputs = inputs.to(device)
                labels = torch.squeeze(labels, 1).long().to(device)
                model = model_test.to(device)
                xx = inputs.cpu().numpy()
                target_labels = (labels + 1) % 10
                # shift all class loops one to the right, 1=>2, 2=>3, .., 9=>0
                for i in range(inputs.shape[0]):
                    min_ele = torch.min(inputs[i])
                    inputs[i] -= min_ele
                    inputs[i] /= torch.max(inputs[i])
                attack2 = torchattacks.BIM(model, epsilon, alpha, steps=10)
                adv_images = attack2(inputs, labels)

                predictions, _ = model_test(adv_images)
                # m = nn.Softmax(dim=1)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                _, predicted = torch.max(predictions, 1)
                # _, predicted = torch.max(predictions, 1)
                # predicted = predictions
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            val_acc_bim = 100 * val_correct / val_total
            print(" Accuracy after BIM adverserial attack %.2f%%" % val_acc_bim)

            # One Pixel Attack
            val_correct = 0
            val_total = 0
            for batch_idx, (inputs, labels) in enumerate(test_loader):

                inputs = inputs.to(device)
                labels = torch.squeeze(labels, 1).long().to(device)
                model = model_test.to(device)
                xx = inputs.cpu().numpy()
                target_labels = (labels + 1) % 10
                # shift all class loops one to the right, 1=>2, 2=>3, .., 9=>0
                for i in range(inputs.shape[0]):
                    min_ele = torch.min(inputs[i])
                    inputs[i] -= min_ele
                    inputs[i] /= torch.max(inputs[i])
                attack2 = torchattacks.Pixle(model, 5, 7)
                adv_images = attack2(inputs, labels)

                predictions, _ = model_test(adv_images)
                # m = nn.Softmax(dim=1)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                _, predicted = torch.max(predictions, 1)
                # predicted = predictions
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            val_acc_bim = 100 * val_correct / val_total
            print(" Accuracy after One Pixel adverserial attack %.2f%%" % val_acc_bim)

            #EOTPGD  Attack
            val_correct = 0
            val_total = 0
            for batch_idx, (inputs, labels) in enumerate(test_loader):

                inputs = inputs.to(device)
                labels = torch.squeeze(labels, 1).long().to(device)
                model = model_test.to(device)
                xx = inputs.cpu().numpy()
                target_labels = (labels + 1) % 10
                # shift all class loops one to the right, 1=>2, 2=>3, .., 9=>0
                for i in range(inputs.shape[0]):
                    min_ele = torch.min(inputs[i])
                    inputs[i] -= min_ele
                    inputs[i] /= torch.max(inputs[i])
                attack2 = torchattacks.EOTPGD(model, epsilon, alpha, steps=10)
                adv_images = attack2(inputs, labels)

                predictions, _ = model_test(adv_images)
                # m = nn.Softmax(dim=1)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                _, predicted = torch.max(predictions, 1)
                # predicted = predictions
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            val_acc_bim = 100 * val_correct / val_total
            print(" Accuracy after EOTPGD adverserial attack %.2f%%" % val_acc_bim)

            # FFGSM Attack

            val_correct = 0
            val_total = 0
            for batch_idx, (inputs, labels) in enumerate(test_loader):

                inputs = inputs.to(device)
                labels = torch.squeeze(labels, 1).long().to(device)
                model = model_test.to(device)
                xx = inputs.cpu().numpy()
                atk = torchattacks.PGD(model, epsilon, alpha, steps=4)
                atk.set_mode_targeted_by_label(quiet=True)  # do not show the message
                target_labels = (labels + 1) % 10
                # shift all class loops one to the right, 1=>2, 2=>3, .., 9=>0
                for i in range(inputs.shape[0]):
                    min_ele = torch.min(inputs[i])
                    inputs[i] -= min_ele
                    inputs[i] /= torch.max(inputs[i])
                attack1 = torchattacks.FFGSM(model, epsilon, alpha)
                adv_images = attack1(inputs, labels)

                predictions, _ = model_test(adv_images)
                # m = nn.Softmax(dim=1)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                _, predicted = torch.max(predictions, 1)
                # predicted = predictions
                # # t = torch.from_numpy(predictions)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            val_acc = 100 * val_correct / val_total

            # PGDRS Attack
            val_correct = 0
            val_total = 0
            for batch_idx, (inputs, labels) in enumerate(test_loader):

                inputs = inputs.to(device)
                labels = torch.squeeze(labels, 1).long().to(device)
                model = model_test.to(device)
                xx = inputs.cpu().numpy()
                target_labels = (labels + 1) % 10
                # shift all class loops one to the right, 1=>2, 2=>3, .., 9=>0
                for i in range(inputs.shape[0]):
                    min_ele = torch.min(inputs[i])
                    inputs[i] -= min_ele
                    inputs[i] /= torch.max(inputs[i])
                attack2 = torchattacks.PGDRSL2(model, eps=8 / 255, alpha=10 / 255)
                adv_images = attack2(inputs, labels)

                predictions,_ = model_test(adv_images)
                # m = nn.Softmax(dim=1)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                _, predicted = torch.max(predictions, 1)
                # predicted = predictions
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            val_acc_bim = 100 * val_correct / val_total
            print(" Accuracy after PGDRS adverserial attack %.2f%%" % val_acc_bim)

            print(" Accuracy after FFGSM adverserial attack %.2f%%" % val_acc)
        else:
            # PGD Attack
            val_correct = 0
            val_total = 0
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                # for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = torch.squeeze(labels, 1).long().to(device)
                atk = torchattacks.PGD(model_test, epsilon, alpha)
                atk.set_mode_targeted_by_label(quiet=True)  # do not show the message
                target_labels = (labels + 1) % 10
                # shift all class loops one to the right, 1=>2, 2=>3, .., 9=>0
                for i in range(inputs.shape[0]):
                    min_ele = torch.min(inputs[i])
                    inputs[i] -= min_ele
                    inputs[i] /= torch.max(inputs[i])
                attack1 = torchattacks.PGD(model_test, epsilon, alpha)
                adv_images = attack1(inputs, labels)
                predictions, _ = model_test(adv_images)
                # m = nn.Softmax(dim=1)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                _, predicted = torch.max(predictions, 1)
                # predicted = predictions
                # t = torch.from_numpy(predictions)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            val_acc_pgd = 100 * val_correct / val_total
            print(" Accuracy after PGD adverserial attack %.2f%%" % val_acc_pgd)

            # FGSM Attack
            val_correct = 0
            val_total = 0
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                # for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = torch.squeeze(labels, 1).long().to(device)
                atk = torchattacks.FFGSM(model_test, epsilon, alpha)
                atk.set_mode_targeted_by_label(quiet=True)  # do not show the message
                target_labels = (labels + 1) % 10
                # shift all class loops one to the right, 1=>2, 2=>3, .., 9=>0
                for i in range(inputs.shape[0]):
                    min_ele = torch.min(inputs[i])
                    inputs[i] -= min_ele
                    inputs[i] /= torch.max(inputs[i])
                attack1 = torchattacks.FFGSM(model_test, epsilon, alpha)
                adv_images = attack1(inputs, labels)
                predictions, _ = model_test(adv_images)
                # m = nn.Softmax(dim=1)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                _, predicted = torch.max(predictions, 1)
                # predicted = predictions
                # t = torch.from_numpy(predictions)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            val_acc_fgsm = 100 * val_correct / val_total
            print(" Accuracy after FGSM adverserial attack %.2f%%" % val_acc_fgsm)
            # BIM Attack
            val_correct = 0
            val_total = 0
            for batch_idx, (inputs, labels) in enumerate(val_loader):

                inputs = inputs.to(device)
                labels = torch.squeeze(labels, 1).long().to(device)
                model = model_test.to(device)
                xx = inputs.cpu().numpy()
                target_labels = (labels + 1) % 10
                # shift all class loops one to the right, 1=>2, 2=>3, .., 9=>0
                for i in range(inputs.shape[0]):
                    min_ele = torch.min(inputs[i])
                    inputs[i] -= min_ele
                    inputs[i] /= torch.max(inputs[i])
                attack2 = torchattacks.BIM(model, epsilon, alpha, steps=10)
                adv_images = attack2(inputs, labels)

                predictions, _ = model_test(adv_images)
                # m = nn.Softmax(dim=1)
                # outputs = m(predictions).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                _, predicted = torch.max(predictions, 1)
                # _, predicted = torch.max(predictions, 1)
                # predicted = predictions
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            val_acc_bim = 100 * val_correct / val_total
            print(" Accuracy after BIM adverserial attack %.2f%%" % val_acc_bim)

         #Checking Cross dataset performance



        fitness = 3*(val_acc_pgd * val_acc_fgsm * val_acc_bim )/val_acc_pgd + val_acc_fgsm + val_acc_bim
        return fitness
    def train_clean(self, model, epochs, hash_indv, grad_clip, evaluation, data_flag, output_root, num_epochs, gpu_ids,
              batch_size, download, run):
        # This is old code
        lr = 0.001
        gamma = 0.1
        milestones = [0.5 * num_epochs, 0.75 * num_epochs]

        info = INFO[data_flag]
        task = info['task']
        n_channels = 3
        n_classes = len(info['label'])

        DataClass = getattr(medmnist, info['python_class'])

        str_ids = gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                gpu_ids.append(id)
        if len(gpu_ids) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])

        device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')

        output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
        if not os.path.exists(output_root):
            os.makedirs(output_root)

        print('==> Preparing data...')

        data_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])])

        train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb)
        val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb)
        test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb)

        train_loader = data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
        train_loader_at_eval = data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
        val_loader = data.DataLoader(dataset=val_dataset,
                                     batch_size=batch_size,
                                     shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)

        print('==> Building and training model...')

        model = model.to(device)

        train_evaluator = medmnist.Evaluator(data_flag, 'train')
        val_evaluator = medmnist.Evaluator(data_flag, 'val')
        test_evaluator = medmnist.Evaluator(data_flag, 'test')

        if task == "multi-label, binary-class":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        train_metrics = self.__test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run, 'train',
                                    output_root)
        val_metrics = self.__test(model, val_evaluator, val_loader, task, criterion, device, run, 'val', output_root)
        test_metrics = self.__test(model, test_evaluator, test_loader, task, criterion, device, run, 'test',
                                   output_root)

        print('train  auc: %.5f  acc: %.5f\n  f1: %.5f\n' % (train_metrics[1], train_metrics[2], train_metrics[3]) + \
              'val  auc: %.5f  acc: %.5f\n f1: %.5f\n' % (val_metrics[1], val_metrics[2], val_metrics[3]) + \
              'test  auc: %.5f  acc: %.5f\n f1: %.5f\n' % (test_metrics[1], test_metrics[2], test_metrics[3]))

        if num_epochs == 0:
            return

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        classifier = PyTorchClassifier(
            model=model,
            loss=criterion,
            optimizer=optimizer,
            input_shape=(3, 28, 28),
            nb_classes=10,
        )

        logs = ['loss', 'auc', 'acc']
        train_logs = ['train_' + log for log in logs]
        val_logs = ['val_' + log for log in logs]
        test_logs = ['test_' + log for log in logs]
        log_dict = OrderedDict.fromkeys(train_logs + val_logs + test_logs, 0)

        writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))

        best_auc = 0
        best_epoch = 0
        best_model = model

        global iteration
        iteration = 0
        # Training the models till the given epochs
        for epoch in trange(num_epochs):
            train_loss = self.__train(model, train_loader, task, criterion, optimizer, device, writer)

            train_metrics = self.__test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run,
                                        'train')
            val_metrics = self.__test(model, val_evaluator, val_loader, task, criterion, device, run, 'val')
            test_metrics = self.__test(model, test_evaluator, test_loader, task, criterion, device, run, 'test')

            scheduler.step()

            for i, key in enumerate(train_logs):
                log_dict[key] = train_metrics[i]
            for i, key in enumerate(val_logs):
                log_dict[key] = val_metrics[i]
            for i, key in enumerate(test_logs):
                log_dict[key] = test_metrics[i]

            for key, value in log_dict.items():
                writer.add_scalar(key, value, epoch)

            cur_auc = val_metrics[1]
            if cur_auc > best_auc:
                best_epoch = epoch
                best_auc = cur_auc
                best_model = model
                print('cur_best_auc:', best_auc)
                print('cur_best_epoch', best_epoch)

        state = {
            'net': best_model.state_dict(),
        }

        path = os.path.join(output_root, 'best_model.pth')
        torch.save(state, path)

        train_metrics = self.__test(best_model, train_evaluator, train_loader_at_eval, task, criterion, device, run,
                                    'train',
                                    output_root)
        val_metrics = self.__test(best_model, val_evaluator, val_loader, task, criterion, device, run, 'val',
                                  output_root)
        test_metrics = self.__test(best_model, test_evaluator, test_loader, task, criterion, device, run, 'test',
                                   output_root)

        train_log = 'train  auc: %.5f  acc: %.5f\n   f1: %.5f\n' % (
            train_metrics[1], train_metrics[2], train_metrics[3])
        val_log = 'val  auc: %.5f  acc: %.5f\n   f1: %.5f\n' % (val_metrics[1], val_metrics[2], train_metrics[3])
        test_log = 'test  auc: %.5f  acc: %.5f\n   f1: %.5f\n' % (test_metrics[1], test_metrics[2], train_metrics[3])

        log = '%s\n' % (data_flag) + train_log + val_log + test_log
        print(log)

        with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
            f.write(log)

        # Step 6: Generate adversarial test examples
        attack = FastGradientMethod(estimator=classifier, eps=0.2)
        attack1 = BasicIterativeMethod(estimator=classifier)
        # attack2 =  PixelAttack(classifier=classifier)
        aa = val_loader.dataset.imgs
        aa = np.swapaxes(aa, 1, 3)
        labels = val_loader.dataset.labels
        x_test_adv = attack.generate(x=aa)

        # Step 7: Evaluate the ART classifier on adversarial test examples
        # targets = torch.squeeze(labels, 1).long().to(device)

        predictions = classifier.predict(x_test_adv)

        if task == 'multi-label, binary-class':
            targets = labels.to(torch.float32).to(device)
            loss = criterion(predictions, targets)
            m = nn.Sigmoid()
            outputs = m(predictions).to(device)
        else:
            targets = torch.from_numpy(labels).to(torch.float32).to(device)
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(torch.from_numpy(predictions).to(device), targets)
            m = nn.Softmax(dim=1)
            t = torch.from_numpy(predictions)
            outputs = m(t).to(device)
            val_evaluator = medmnist.Evaluator(self.medmnist_dataset, 'val')
            y_score = outputs.detach().cpu().numpy()
            auc, acc = val_evaluator.evaluate(y_score, output_root, run)
            print(loss)
            print(auc)
            print(acc)
            # m = nn.Softmax(dim=1)
            # outputs = m(predictions).to(device)
            # targets = targets.float().resize_(len(targets), 1)

        writer.close()
        fitness = pgd_acc + acc + val_acc_bim / 3
        return
        if evaluation == 'valid':
            return 1 - val_metrics[3]
        else:
            return 1 - test_metrics[3]

