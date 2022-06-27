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
#import torch_xla
#import torch_xla.core.xla_model as xm

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
from tqdm import tqdm

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

    print(Fore.GREEN + 'Baseline:', args.baseline_type)
    print(Fore.GREEN + 'Frame aggregation method:', args.frame_aggregation)
    print(Fore.GREEN + 'Current architecture:', args.arch)
    print(Fore.GREEN + 'Num class:', args.num_class)
    print(Fore.GREEN + 'target data usage:', args.use_target)
    if args.use_target == 'none':
        print(Fore.GREEN + 'no Domain Adaptation')
    else:
        if args.dis_DA != 'none':
            print(Fore.GREEN + 'Apply the discrepancy-based Domain Adaptation approach:', args.dis_DA)
            if len(args.place_dis) != args.add_fc + 2:
                raise ValueError(Back.RED + 'len(place_dis) should be equal to add_fc + 2')

        if args.adv_DA != 'none':
            print(Fore.GREEN + 'Apply the adversarial-based Domain Adaptation approach:', args.adv_DA)

        if args.use_bn != 'none':
            print(Fore.GREEN + 'Apply the adaptive normalization approach:', args.use_bn)

    print(Fore.YELLOW + 'Current modality:', args.modality)
    print(Fore.YELLOW + 'From dataset', args.source_domain, Fore.YELLOW + 'to dataset', args.target_domain)

    print("------------------------------")
    print(Fore.GREEN + 'val list:', args.val_list)
    print(Fore.GREEN + 'train source list:', args.train_source_list)
    print(Fore.GREEN + 'train target list:', args.train_target_list)
    print(Fore.GREEN + 'val data:', args.val_data)
    print(Fore.GREEN + 'train source data:', args.train_source_data)
    print(Fore.GREEN + 'train target data:', args.train_target_data)
    print("------------------------------")

    # === Setup TPU device ===#
    dev = torch.device("cuda:0")

    # determine the categories
    # want to allow multi-label classes.

    # Original way to compute number of classes
    ####class_names = [line.strip().split(' ', 1)[1] for line in open(args.class_file)]
    ####num_class = len(class_names)

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
    path_exp = args.exp_path + args.modality + '/'
    if not os.path.isdir(path_exp):
        os.makedirs(path_exp)

    note_fl = open(path_exp + '/note.txt', 'w')
    note_fl.write('Number of Epochs = {}\n'
                  'lr = {}\n'
                  'Train Batch Size = {}\n'
                  'Sequence Length = {}\n'
                  'Decay steps = {}\n'
                  'Decay factor = {}\n'
                  'Memory size = {}\n'
                  'Memory cam classes = {}\n'.format(args.epochs, args.lr, args.batch_size, args.num_segments, args.lr_steps, args.lr_decay,
                                                     args.mem_size, args.outPool_size))

    note_fl.close()

    # Log files
    writer = SummaryWriter(path_exp)
    train_log_loss = open((path_exp + '/train_log_loss.txt'), 'w')
    train_log_acc = open((path_exp + '/train_log_acc.txt'), 'w')
    train_log_loss_batch = open((path_exp + '/train_log_loss_batch.txt'), 'w')
    test_log_loss = open((path_exp + '/test_log_loss.txt'), 'w')
    test_log_acc = open((path_exp + '/test_log_acc.txt'), 'w')

    # === initialize the model ===#
    print(Fore.CYAN + 'preparing the model......')

    cudnn.benchmark = True

    # --- open log files ---#
    result_file = open(path_exp + 'result_file.log', 'w')

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

    print(Fore.CYAN + 'Number of train samples = {}'.format(source_set.__len__()))

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

    print(Fore.CYAN + 'Number of test samples = {}'.format(target_set.__len__()))

    target_sampler = torch.utils.data.sampler.RandomSampler(target_set)
    target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size[1], shuffle=False,
                                                sampler=target_sampler, num_workers=args.workers, pin_memory=True)

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
    model.to(dev)

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
        print(Fore.GREEN + 'Epoch {}'.format(epoch))

        optim_scheduler.step()
        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        model.classifier.train(True)
        writer.add_scalar('lr', optimizer_fn.param_groups[0]['lr'])
        print(Fore.CYAN + 'Learning rate: {}'.format(optimizer_fn.param_groups[0]['lr']))
        data_loader = enumerate(zip(source_loader, target_loader))

        for i, ((source_data, source_label, source_id), (target_data, target_label, target_id)) in data_loader:
            if source_data.size(0) != source_label.size(0):
              print('Skipped for different size: {} {}'.format(source_data.size(0), source_label.size(0)))
              continue
            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()
            trainSamples += source_data.size(2) #forse .size(1)
            sourceVariable = source_data.permute(1, 0, 2, 3, 4)
            output_label, _ = model(sourceVariable.to(dev))
            loss = loss_fn(output_label.to(dev), source_label.to(dev))
            loss.backward()
            optimizer_fn.step()
            _, predicted = torch.max(output_label.data, 1)
            numCorrTrain += (predicted == source_label.to(dev)).sum()
            loss_value = loss.item()  # loss.data[0]
            epoch_loss += loss_value
            if train_iter%10 == 0:
                line = 'Training loss after {} iterations = {} '.format(train_iter, loss_value)
                print(line)
                result_file.write(line + '\n')
                train_log_loss_batch.write('Training loss after {} iterations = {}\n'.format(train_iter, loss_value))
                writer.add_scalar('train/iter_loss', loss_value, train_iter)
        avg_loss = epoch_loss / iterPerEpoch
        trainAccuracy = (numCorrTrain / trainSamples) * 100
        line_avg = 'Average training loss after {} epoch = {} '.format(epoch, avg_loss)
        line_acc = 'Training accuracy after {} epoch = {}% '.format(epoch, trainAccuracy)
        print(line_avg)
        print(line_acc)
        result_file.write(line_avg + '\n' + line_acc + '\n')
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        writer.add_scalar('train/accuracy', trainAccuracy, epoch)
        train_log_loss.write('Training loss after {} epoch = {}\n'.format(epoch, avg_loss))
        train_log_acc.write(line_acc)

        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            print('Testing...')
            model.train(False)
            test_loss_epoch = 0
            test_iter = 0
            test_samples = 0
            numCorr = 0
            for i, (val_data, val_label, _) in enumerate(target_loader):
                test_iter += 1
                test_samples += val_data.size(0)
                valdataVariable = val_data.permute(1, 0, 2, 3, 4)
                output_label, _ = model(valdataVariable.to(dev))
                test_loss = loss_fn(output_label.to(dev), val_label.to(dev))
                test_loss_epoch += test_loss.item()
                _, predicted = torch.max(output_label.data, 1)
                numCorr += (predicted == val_label.to(dev)).sum()
            test_accuracy = (numCorr / test_samples) * 100
            avg_test_loss = test_loss_epoch / test_iter
            line_avg = 'Test Loss after {} epochs, loss = {}'.format(epoch, avg_test_loss)
            line_acc = 'Test Accuracy after {} epochs = {}%'.format(epoch, test_accuracy)
            print(line_avg)
            print(line_acc)
            result_file.write(line_avg + '\n' + line_acc + '\n')
            writer.add_scalar('test/epoch_loss', avg_test_loss, epoch)
            writer.add_scalar('test/accuracy', test_accuracy, epoch)
            test_log_loss.write(line_avg)
            test_log_acc.write(line_acc)

            if test_accuracy > best_acc:
                best_acc = test_accuracy
                print(Fore.RED + 'UPDATE BEST ACCURACY:', best_acc)

    end_train = time.time()
    print(Fore.CYAN + 'total training time:', end_train - start_train)

    # --- write the total time to log files ---#
    line_time = 'total time: {:.3f} '.format(end_train - start_train)
    print(line_time)
    result_file.write(line_time + '\n')

    # --- close log files ---#
    result_file.close()
    train_log_loss.close()
    train_log_acc.close()
    test_log_acc.close()
    train_log_loss_batch.close()
    test_log_loss.close()
    writer.export_scalars_to_json(path_exp + "/all_scalars.json")
    writer.close()

if __name__ == '__main__':
    main()
