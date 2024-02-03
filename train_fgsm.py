import argparse
import copy
import logging
import os
import time

import PIL
import torch.utils.data as data
from art.estimators.classification import PyTorchClassifier

import torchattacks
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from preact_resnet import PreActResNet18
from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
                   attack_pgd, evaluate_pgd, evaluate_standard, evaluate_fgsm)
import medmnist
from medmnist import INFO, Evaluator
logger = logging.getLogger(__name__)
from attacks import fgsm_attack, rfgsm_attack, fab_attack, apgd_attack, apgdt_attack, pgd_attack, square_attack


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    return parser.parse_args()


def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, download, model_flag, resize, as_rgb, model_path, run):
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

    if resize:
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
             transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])])
    else:
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

    train_evaluator = medmnist.Evaluator(data_flag, 'train')
    val_evaluator = medmnist.Evaluator(data_flag, 'val')
    test_evaluator = medmnist.Evaluator(data_flag, 'test')

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()


    args = get_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, 'output.log'))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)


    mean = 0.
    std = 0.
    for images, _ in train_loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(train_loader.dataset)
    std /= len(train_loader.dataset)

    # if channels==1:
    #     mu = torch.tensor(mean).view(1, 1, 1).cuda()
    #     std = torch.tensor(std).view(1, 1, 1).cuda()
    # else:
    mu = torch.tensor(mean).view(3, 1, 1).cuda()
    std = torch.tensor(std).view(3, 1, 1).cuda()

    # upper_limit = ((1 - mu) / std)
    # lower_limit = ((0 - mu) / std)
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    pgd_alpha = (2 / 255.) / std



    model = PreActResNet18(n_classes).cuda()
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    #model, opt = amp.initialize(model, opt, **amp_args)
    #criterion = nn.CrossEntropyLoss()

    if args.delta_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Training
    prev_robust_acc = 0.
    start_train_time = time.time()
    print('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            if i == 0:
                first_batch = (X, y)
            if args.delta_init != 'previous':
                delta = torch.zeros_like(X).cuda()
            if args.delta_init == 'random':
                for j in range(len(epsilon)):
                    delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True
            # Enables autocasting for the forward pass (model + loss)
            with amp.autocast():
                output = model(X + delta[:X.size(0)])
                if task == 'multi-label, binary-class':
                    y = y.to(torch.float32).to(device)
                    loss = criterion(output, y)
                else:
                    y = torch.squeeze(y, 1).long().to(device)
                    loss = criterion(output, y)

            # with amp.scale_loss(loss, opt) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()
            grad = delta.grad.detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            delta = delta.detach()
            with amp.autocast():
                output = model(X + delta[:X.size(0)])
                loss = criterion(output, y)
            opt.zero_grad()
            #with amp.scale_loss(loss, opt) as scaled_loss:
                #scaled_loss.backward()
            loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()
        if args.early_stop:
            # Check current PGD robustness of model using random minibatch
            X, y = first_batch
            pgd_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 5, 1, opt)
            with torch.no_grad():
                output = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
            robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
            if robust_acc - prev_robust_acc < -0.2:
                break
            prev_robust_acc = robust_acc
            best_state_dict = copy.deepcopy(model.state_dict())
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        print('Epoch ',epoch,' Epoch Time ', epoch_time - start_epoch_time,' Learning Rate  ' ,lr,' Loss ', train_loss/train_n,' Train Accuracy', train_acc/train_n)
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
    train_time = time.time()
    if not args.early_stop:
        best_state_dict = model.state_dict()
    torch.save(best_state_dict, os.path.join(args.out_dir, 'model.pth'))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    model_test = PreActResNet18(n_classes).cuda()
    model_test.load_state_dict(best_state_dict)
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test,criterion, 7, 10,task,device,std)
    fgsm_loss, fgsm_acc = evaluate_fgsm(test_loader, model_test, criterion, 7, 10, task, device, std)
    #pgd_loss, pgd_acc = evaluate_fgsm(test_loader, model_test, criterion, 50, 10, task, device, std)
    test_loss, test_acc = evaluate_standard(test_loader, model_test,criterion,task,device,std)






    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
    print('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    print('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
    print('Test Loss \t Test Acc \t FGSM Loss \t FGSM Acc')
    print('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, fgsm_loss, fgsm_acc)
    alpha = (2 / 255.) / std

    #Check the adverserial attacks on the trained model
    acc = fgsm_attack(model_test,test_loader,epsilon)
    print("Accuracy of FGSM attack",acc)
    acc = square_attack(model_test,test_loader,2/255)
    print("Accuracy of Square attack",acc)

    classifier = PyTorchClassifier(
        model=model_test,
        loss=criterion,
        optimizer=opt,
        input_shape=(3, 28, 28),
        nb_classes=n_classes,
    )
    # FGSM Attack
    val_correct = 0
    val_total = 0
    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = torch.squeeze(labels, 1).long().to(device)
        model = model_test.to(device)
        atk = torchattacks.FFGSM(model_test, eps=8 / 255, alpha=2 / 255)
        atk.set_mode_targeted_by_label(quiet=True)  # do not show the message
        target_labels = (labels + 1) % 10
        # shift all class loops one to the right, 1=>2, 2=>3, .., 9=>0
        for i in range(inputs.shape[0]):
            min_ele = torch.min(inputs[i])
            inputs[i] -= min_ele
            inputs[i] /= torch.max(inputs[i])
        attack1 = torchattacks.FFGSM(model, eps=8 / 255, alpha=10 / 255)
        adv_images = attack1(inputs, labels)
        predictions = model_test(adv_images)
        m = nn.Softmax(dim=1)
        #t = torch.from_numpy(predictions)
        outputs = m(predictions).to(device)
        _, predicted = torch.max(outputs.data, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()
    val_acc = 100 * val_correct / val_total
    print(" Accuracy after FGSM adverserial attack %.2f%%" % val_acc)
    logging.info("Accuracy after FGSM adverserial attack %.2f%%" % val_acc)
    # BIM Attack
    val_correct = 0
    val_total = 0
    for inputs, labels in test_loader:

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
        attack2 = torchattacks.BIM(model, eps=8 / 255, alpha=2 / 255, steps=10)
        adv_images = attack2(inputs, labels)

        predictions = model_test(adv_images)
        m = nn.Softmax(dim=1)
        outputs = m(predictions).to(device)
        _, predicted = torch.max(outputs.data, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()
    val_acc = 100 * val_correct / val_total
    print(" Accuracy after BIM adverserial attack %.2f%%" % val_acc)
    logging.info("Accuracy after BIM adverserial attack %.2f%%" % val_acc)

    # FFGSM Attack

    val_correct = 0
    val_total = 0
    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = torch.squeeze(labels, 1).long().to(device)
        model = model_test.to(device)
        xx = inputs.cpu().numpy()
        atk = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=4)
        atk.set_mode_targeted_by_label(quiet=True)  # do not show the message
        target_labels = (labels + 1) % 10
        # shift all class loops one to the right, 1=>2, 2=>3, .., 9=>0
        for i in range(inputs.shape[0]):
            min_ele = torch.min(inputs[i])
            inputs[i] -= min_ele
            inputs[i] /= torch.max(inputs[i])
        attack1 = torchattacks.FFGSM(model, eps=8 / 255, alpha=10 / 255)
        adv_images = attack1(inputs, labels)

        predictions = model(adv_images)
        m = nn.Softmax(dim=1)
        # t = torch.from_numpy(predictions)
        outputs = m(predictions).to(device)
        _, predicted = torch.max(outputs.data, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()
    val_acc = 100 * val_correct / val_total
    print(" Accuracy after FFGSM adverserial attack %.2f%%" % val_acc)
    logging.info("Accuracy after FFGSM adverserial attack %.2f%%" % val_acc)


if __name__ == "__main__":
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(
            description='RUN Baseline model of MedMNIST2D')

        parser.add_argument('--data_flag',
                            default='organcmnist',
                            type=str)
        parser.add_argument('--output_root',
                            default='./output',
                            help='output root, where to save models and results',
                            type=str)
        parser.add_argument('--num_epochs',
                            default=20,
                            help='num of epochs of training, the script would only test model if set num_epochs to 0',
                            type=int)
        parser.add_argument('--gpu_ids',
                            default='0',
                            type=str)
        parser.add_argument('--batch_size',
                            default=64,
                            type=int)
        parser.add_argument('--download',
                            action="store_true")
        parser.add_argument('--resize',
                            help='resize images of size 28x28 to 224x224',
                            action="store_true",
                            default=False)
        parser.add_argument('--as_rgb',
                            help='convert the grayscale image to RGB',
                            default=True,
                            action="store_true")
        parser.add_argument('--model_path',
                            default=None,
                            help='root of the pretrained model to test',
                            type=str)
        parser.add_argument('--model_flag',
                            default='vgg16',
                            help='choose backbone from resnet18, resnet50',
                            type=str)
        parser.add_argument('--run',
                            default='model1',
                            help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                            type=str)

        args = parser.parse_args()
        data_flag = args.data_flag
        output_root = args.output_root
        num_epochs = args.num_epochs
        gpu_ids = args.gpu_ids
        batch_size = args.batch_size
        download = args.download
        model_flag = args.model_flag
        resize = args.resize
        as_rgb = args.as_rgb
        model_path = args.model_path
        run = args.run
    main(data_flag, output_root, num_epochs, gpu_ids, batch_size, download, model_flag, resize, as_rgb, model_path, run)
