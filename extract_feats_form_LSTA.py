# import argparse
import os
import time
import shutil
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

from LSTA.attentionModule import attentionModel
from dataset import TSNDataSet, TSNSpatialDataSet
from models import VideoModel
from loss import *
# from opts import parser
from opts_fixed import parser
from utils.utils import randSelectBatch
import math
import pandas as pd

from colorama import init
from colorama import Fore, Back, Style
import numpy as np
from tensorboardX import SummaryWriter

from pathlib import Path

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

init(autoreset=True)

# best_prec1 = 0
gpu_count = torch.cuda.device_count()


def main():
    global args, writer_train, writer_val

    best_prec1 = 0
    args = parser.parse_args()

    # New approach
    num_class_str = args.num_class.split(",")
    # single class
    if len(num_class_str) < 1:
        raise Exception("Must specify a number of classes to train")
    else:
        num_class = []
        for num in num_class_str:
            num_class.append(int(num))

    # === check the folder existence ===#

    # === initialize the model ===#
    print(Fore.CYAN + 'preparing the model......')

    cudnn.benchmark = True

    # --- open log files ---#

    # === Model ===#
    train_params = []
    best_acc = 0

    model = attentionModel(num_classes=num_class[0], mem_size=args.mem_size, c_cam_classes=args.outPool_size)
    model.train(False)
    for params in model.parameters():
        params.requires_grad = False
    for params in model.lsta_cell.parameters():
        params.requires_grad = True
        train_params += [params]
    for params in model.classifier.parameters():
        params.requires_grad = True
        train_params += [params]

    model.classifier.train(True)
    model.cpu()

    for name, param in model.named_parameters():
        if name in ['fc.weight', 'fc.bias']:
            param.requires_grad = True
        #else:
        #    param.requires_grad = False

    for name, param in model.named_parameters():
        print(name, ':', param.requires_grad)
        print('param.shape: ', param.shape)
        print('=====')

    for name, child in model.named_children():
        print('name: ', name)
        print('isinstance({}, nn.Module): '.format(name), isinstance(child, nn.Module))
        print('=====')

    for key in model.fc.state_dict():
        print('key: ', key)
        param = model.fc.state_dict()[key]
        print('param.shape: ', param.shape)
        print('param.requires_grad: ', param.requires_grad)
        print('param.shape, param.requires_grad: ', param.shape, param.requires_grad)
        print('isinstance(param, nn.Module) ', isinstance(param, nn.Module))
        print('isinstance(param, nn.Parameter) ', isinstance(param, nn.Parameter))
        print('isinstance(param, torch.Tensor): ', isinstance(param, torch.Tensor))
        print('=====')

    print(model.state_dict())

    exit(0)

    # === Data loading ===#
    print(Fore.CYAN + 'loading data......')

    if args.use_opencv:
        print("use opencv functions")

    if args.modality == 'Audio' or 'RGB' or args.modality == 'ALL':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus']:
        data_length = 1

    # calculate the number of videos to load for training in each list ==> make sure the iteration # of source & target are same
    num_source = len(pd.read_pickle(args.train_source_list).index)
    num_target = len(pd.read_pickle(args.train_target_list).index)
    num_val = len(pd.read_pickle(args.val_list).index)

    num_iter_source = num_source / args.batch_size[0]
    num_iter_target = num_target / args.batch_size[1]
    num_max_iter = max(num_iter_source, num_iter_target)
    num_source_train = round(num_max_iter * args.batch_size[0]) if args.copy_list[0] == 'Y' else num_source
    num_target_train = round(num_max_iter * args.batch_size[1]) if args.copy_list[1] == 'Y' else num_target

    train_source_data = Path(args.train_source_data_spatial + ".hkl")
    train_source_list = Path(args.train_source_list)
    source_set = TSNSpatialDataSet(train_source_data, train_source_list,
                            num_dataload=num_source_train,
                            num_segments=args.num_segments,
                            total_segments=5,
                            new_length=data_length, modality=args.modality,
                            image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2",
                                                                            "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
                            random_shift=False,
                            test_mode=True
                            )

    source_sampler = torch.utils.data.sampler.RandomSampler(source_set)
    source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size[0], shuffle=False,
                                                sampler=source_sampler, num_workers=args.workers, pin_memory=True)

    train_target_data = Path(args.train_target_data_spatial + ".hkl")
    train_target_list = Path(args.train_target_list)
    target_set = TSNSpatialDataSet(train_target_data, train_target_list,
                            num_dataload=num_target_train,
                            num_segments=args.num_segments,
                            total_segments=5,
                            new_length=data_length, modality=args.modality,
                            image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2",
                                                                            "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
                            random_shift=False,
                            test_mode=True,
                            )

    target_sampler = torch.utils.data.sampler.RandomSampler(target_set)
    target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size[1], shuffle=False,
                                                sampler=target_sampler, num_workers=args.workers, pin_memory=True)

    # === Optimizer === #
    loss_fn = nn.CrossEntropyLoss()
    optimizer_fn = torch.optim.Adam(train_params, lr=args.lr, weight_decay=5e-4, eps=1e-4)
    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_fn, milestones=args.lr_steps, gamma=args.lr_decay)
    train_iter = 0

    # === Training ===#
    start_train = time.time()
    print(Fore.CYAN + 'start training......')

    start_epoch = 1

    for epoch in range(start_epoch, args.epochs + 1):

        optim_scheduler.step()
        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        model.classifier.train(True)
        data_loader = enumerate(zip(source_loader, target_loader))

        for i, ((source_data, source_label, source_id), (target_data, target_label, target_id)) in data_loader:
            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()
            trainSamples += source_data.size(2)
            sourceVariable = source_data.permute(1, 0, 2, 3, 4)
            output_label, features_avgpool = model(sourceVariable)
            loss = loss_fn(output_label, target_label)
            loss.backward()
            optimizer_fn.step()
            _, predicted = torch.max(output_label.data, 1)
            numCorrTrain += (predicted == target_label.cpu()).sum()
            loss_value = loss.item()  # loss.data[0]
            epoch_loss += loss_value
            if train_iter%1 == 0:
                print('Training loss after {} iterations = {} '.format(train_iter, loss_value))
                print(Fore.YELLOW + 'Average training loss after {} epoch = {} '.format(epoch + 1, (epoch_loss / iterPerEpoch)))
                print(Fore.YELLOW + 'Training accuracy after {} epoch = {}% '.format(epoch + 1, (numCorrTrain / trainSamples) * 100))
        avg_loss = epoch_loss / iterPerEpoch
        trainAccuracy = (numCorrTrain / trainSamples) * 100
        print('Average training loss after {} epoch = {} '.format(epoch + 1, avg_loss))
        print('Training accuracy after {} epoch = {}% '.format(epoch + 1, trainAccuracy))

        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            print('Testing...')
            model.train(False)
            test_loss_epoch = 0
            test_iter = 0
            test_samples = 0
            numCorr = 0
            for i, (val_data, val_label, _) in enumerate(target_data):
                print('testing inst = {}'.format(i))
                test_iter += 1
                test_samples += val_data.size(0)
                output_label, _ = model(val_data)
                test_loss = loss_fn(output_label, val_label)
                test_loss_epoch += test_loss.data[0]
                _, predicted = torch.max(output_label.data, 1)
                numCorr += (predicted == val_label.cpu()).sum()
            test_accuracy = (numCorr / test_samples) * 100
            avg_test_loss = test_loss_epoch / test_iter
            print('Test Loss after {} epochs, loss = {}'.format(epoch + 1, avg_test_loss))
            print('Test Accuracy after {} epochs = {}%'.format(epoch + 1, test_accuracy))

            if test_accuracy > best_acc:
                best_acc = test_accuracy
                print(Fore.RED + 'UPDATE BEST ACCURACY:', best_acc)

    end_train = time.time()
    print(Fore.CYAN + 'total training time:', end_train - start_train)

    # --- write the total time to log files ---#
    line_time = 'total time: {:.3f} '.format(end_train - start_train)

    # train_file.write(line_time)
    # train_short_file.write(line_time)

    # --- close log files ---#

    if args.tensorboard:
        writer_train.close()
        writer_val.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
