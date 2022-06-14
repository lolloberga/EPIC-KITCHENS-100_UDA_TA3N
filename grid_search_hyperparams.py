import os
import time
import shutil
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from dataset import TSNDataSet
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

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler


def train_action_recognition(config, checkpoint_dir=None, data_dir=None):
    args = parser.parse_args()
    num_class_str = args.num_class.split(",")
    if len(num_class_str) < 1:
        raise Exception("Must specify a number of classes to train")
    else:
        num_class = []
        for num in num_class_str:
            num_class.append(int(num))

    # Init neural model
    model = VideoModel(num_class, args.baseline_type, args.frame_aggregation, args.modality,
                       train_segments=args.num_segments, val_segments=args.val_segments,
                       base_model=args.arch, path_pretrained=args.pretrained,
                       add_fc=args.add_fc, fc_dim=args.fc_dim,
                       dropout_i=args.dropout_i, dropout_v=args.dropout_v, partial_bn=not args.no_partialbn,
                       use_bn=args.use_bn if args.use_target != 'none' else 'none',
                       ens_DA=args.ens_DA if args.use_target != 'none' else 'none',
                       n_rnn=args.n_rnn, rnn_cell=args.rnn_cell, n_directions=args.n_directions, n_ts=args.n_ts,
                       use_attn=args.use_attn, n_attn=args.n_attn, use_attn_frame=args.use_attn_frame,
                       verbose=args.verbose, share_params=args.share_params)
    model = torch.nn.DataParallel(model, args.gpus).cpu()

    # Add optimizer
    if args.optimizer == 'SGD':
        print(Fore.YELLOW + 'using SGD')
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer == 'Adam':
        print(Fore.YELLOW + 'using Adam')
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        print(Back.RED + 'optimizer not support or specified!!!')
        exit()

    # Load data
    source_set, target_set = load_data(data_dir)

    source_sampler = torch.utils.data.sampler.RandomSampler(source_set)
    source_loader = torch.utils.data.DataLoader(source_set, batch_size=config.batch_size, shuffle=False,
                                                sampler=source_sampler, num_workers=args.workers, pin_memory=True)
    target_sampler = torch.utils.data.sampler.RandomSampler(target_set)
    target_loader = torch.utils.data.DataLoader(target_set, batch_size=config.batch_size, shuffle=False,
                                                sampler=target_sampler, num_workers=args.workers, pin_memory=True)


    # Define loss function
    criterion = torch.nn.CrossEntropyLoss().cpu()
    criterion_domain = torch.nn.CrossEntropyLoss().cpu()

    # Training
    for epoch in range(1, args.epochs + 1):
        print()

'''Return both train and test set'''
def load_data(data_dir="./data"):
    return (0, 0)





if __name__ == '__main__':
    train_action_recognition()