import argparse

'''
LEGENDA:
* LIST      = LABELS     (train_val)
* DATA      = FEATURES   (prextracted_model_features)
* SOURCE    = train
* TARGET    = test
'''


class OptsParser:
    def __init__(self, param):
        ego_path = "/home/pol/Desktop/EGO_Project/"
        epic_path = "/home/pol/Desktop/EPIC-KITCHENS-100_UDA_TA3N/"

        CURRENT_DOMAIN = param[0]
        TARGET_DOMAIN = param[1]
        FRAME_AGGREGATION = param[2]
        CURRENT_MODALITY = "RGB"
        USE_TARGET = "uSv"
        CURRENT_ARCH = "tsm"

        N_EPOCH = 50
        DROP = 0.8
        LEARNING = 3e-2
        BATCH = [32,28, 64]
        OPTIMIZ = 'SGD'
        LRN_DECAY = 'noob'
        LRN_ADPT = 'dann'
        LRN_STEP = list(range(10,N_EPOCH,10))
        LRN_DECAY_WEIGHT = 1e-4

        RES = False

        # Used only during DA
        PLACE_ADV = param[3]
        USE_ATTN = param[4]
        ADV_DA = 'none' if PLACE_ADV == ['N', 'N', 'N'] else 'RevGrad'
        LOSS_ATTN = 'none' if USE_ATTN == 'none' else 'attentive_entropy'


        self.parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
        self.parser.add_argument('--source_domain', type=str, default=CURRENT_DOMAIN)
        self.parser.add_argument('--target_domain', type=str, default=TARGET_DOMAIN)

        self.parser.add_argument('--num_class', type=str, default="8,8")  # 97,300
        self.parser.add_argument('--modality', type=str, default=CURRENT_MODALITY)
        # choices=['Audio', 'RGB', 'Flow', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus', 'ALL'])
        self.parser.add_argument('--train_source_list', type=str,
                            # default="I:/Datasets/EgoAction/EPIC-100/annotations/labels_train_test/val/EPIC_100_uda_source_train.pkl")
                            default=ego_path + "train_val/" + CURRENT_DOMAIN + "_train.pkl")
        self.parser.add_argument('--train_target_list', type=str,
                            # default="I:/Datasets/EgoAction/EPIC-100/annotations/labels_train_test/val/EPIC_100_uda_target_train_timestamps.pkl")
                            default=ego_path + "train_val/" + TARGET_DOMAIN + "_train.pkl")  # CURRENT
        self.parser.add_argument('--val_list', type=str,
                            # default="I:/Datasets/EgoAction/EPIC-100/annotations/labels_train_test/val/EPIC_100_uda_target_test_timestamps.pkl")
                            default=ego_path + "train_val/" + TARGET_DOMAIN + "_test.pkl")  # CURRENT
        self.parser.add_argument('--val_data', type=str,
                            # default="I:/Datasets/EgoAction/EPIC-100/frames_rgb_flow/feature/target_val")
                            default=ego_path + "prextracted_model_features/" + CURRENT_MODALITY + "/ek_" +
                                    CURRENT_ARCH + "/" + CURRENT_DOMAIN + "-" + TARGET_DOMAIN + "_test")
        self.parser.add_argument('--train_source_data', type=str,
                            # default="I:/Datasets/EgoAction/EPIC-100/frames_rgb_flow/feature/source_val")
                            default=ego_path + "prextracted_model_features/" + CURRENT_MODALITY + "/ek_" +
                                    CURRENT_ARCH + "/" + CURRENT_DOMAIN + "-" + CURRENT_DOMAIN + "_train")
        self.parser.add_argument('--train_target_data', type=str,
                            # default="I:/Datasets/EgoAction/EPIC-100/frames_rgb_flow/feature/target_val")
                            default=ego_path + "prextracted_model_features/" + CURRENT_MODALITY + "/ek_" +
                                    CURRENT_ARCH + "/" + CURRENT_DOMAIN + "-" + TARGET_DOMAIN + "_train")

        # ========================= Model Configs ==========================
        self.parser.add_argument('--train_metric', default="verb", type=str)
        self.parser.add_argument('--dann_warmup', default=False, action="store_true")
        self.parser.add_argument('--arch', type=str, default=CURRENT_ARCH.upper(), choices=["TBN", "I3D", "TSM"])
        self.parser.add_argument('--pretrained', type=str, default="none")
        self.parser.add_argument('--num_segments', type=int, default=5)
        self.parser.add_argument('--val_segments', type=int, default=5)
        self.parser.add_argument('--add_fc', default=1, type=int, metavar='M',
                            help='number of additional fc layers (excluding the last fc layer) (e.g. 0, 1, 2, ...)')
        self.parser.add_argument('--fc_dim', type=int, default=512, help='dimension of added fc')
        self.parser.add_argument('--baseline_type', type=str, default='video',
                            choices=['frame', 'video', 'tsn'])
        self.parser.add_argument('--frame_aggregation', type=str, default=FRAME_AGGREGATION,
                            choices=['avgpool', 'rnn', 'temconv', 'trn', 'trn-m', 'none'],
                            help='aggregation of frame features (none if baseline_type is not video)')
        self.parser.add_argument('--optimizer', type=str, default=OPTIMIZ, choices=['SGD', 'Adam'])
        self.parser.add_argument('--use_opencv', default=False, action="store_true",
                            help='whether to use the opencv transformation')
        self.parser.add_argument('--dropout_i', '--doi', default=0.5, type=float,
                            metavar='DOI', help='dropout ratio for frame-level feature (default: 0.5)')
        self.parser.add_argument('--dropout_v', '--dov', default=DROP, type=float,
                            metavar='DOV', help='dropout ratio for video-level feature (default: 0.5)')
        self.parser.add_argument('--loss_type', type=str, default="null",
                            choices=['null'])
        self.parser.add_argument('--weighted_class_loss', type=str, default='N', choices=['Y', 'N'])

        # ------ RNN ------
        self.parser.add_argument('--n_rnn', default=1, type=int, metavar='M',
                            help='number of RNN layers (e.g. 0, 1, 2, ...)')
        self.parser.add_argument('--rnn_cell', type=str, default='LSTM', choices=['LSTM', 'GRU'])
        self.parser.add_argument('--n_directions', type=int, default=1, choices=[1, 2],
                            help='(bi-) direction RNN')
        self.parser.add_argument('--n_ts', type=int, default=5, help='number of temporal segments')

        # ========================= DA Configs ==========================
        self.parser.add_argument('--share_params', type=str, default='Y', choices=['Y', 'N'])
        self.parser.add_argument('--use_target', type=str, default=USE_TARGET, choices=['none', 'Sv', 'uSv'],
                            help='the method to use target data (not use | supervised | unsupervised)')
        self.parser.add_argument('--dis_DA', type=str, default='none', choices=['none', 'DAN', 'JAN', 'CORAL'],
                            help='discrepancy method for DA')
        self.parser.add_argument('--adv_DA', type=str, default=ADV_DA, choices=['none', 'RevGrad'],
                            help='adversarial method for DA')
        self.parser.add_argument('--use_bn', type=str, default='none', choices=['none', 'AdaBN', 'AutoDIAL'],
                            help='normalization-based methods')
        self.parser.add_argument('--ens_DA', type=str, default='none', choices=['none', 'MCD'],
                            help='ensembling-based methods')
        self.parser.add_argument('--use_attn_frame', type=str, default='none',
                            choices=['none', 'TransAttn', 'general', 'DotProduct'],
                            help='attention-mechanism for frames only')
        self.parser.add_argument('--use_attn', type=str, default=USE_ATTN,
                            choices=['none', 'TransAttn', 'general', 'DotProduct'],
                            help='attention-mechanism')
        self.parser.add_argument('--n_attn', type=int, default=1, help='number of discriminators for transferable attention')
        self.parser.add_argument('--add_loss_DA', type=str, default=LOSS_ATTN,
                            choices=['none', 'target_entropy', 'attentive_entropy'],
                            help='add more loss functions for DA')
        self.parser.add_argument('--pred_normalize', type=str, default='N', choices=['Y', 'N'])
        self.parser.add_argument('--alpha', default=0, type=float, metavar='M',
                            help='weighting for the discrepancy loss (use scheduler if < 0)')
        self.parser.add_argument('--beta', default=[0.75, 0.75, 0.5], type=float, nargs="+", metavar='M',
                            help='weighting for the adversarial loss (use scheduler if < 0; [relation-beta, video-beta, frame-beta])')
        self.parser.add_argument('--gamma', default=0.003, type=float, metavar='M', # default = 0.3
                            help='weighting for the entropy loss')
        self.parser.add_argument('--mu', default=0, type=float, metavar='M',
                            help='weighting for ensembling loss (e.g. discrepancy)')
        self.parser.add_argument('--weighted_class_loss_DA', type=str, default='N', choices=['Y', 'N'])
        self.parser.add_argument('--place_dis', default=['N', 'Y', 'N'], type=str, nargs="+",
                            metavar='N', help='where to place the discrepancy loss (length = add_fc + 2)')
        self.parser.add_argument('--place_adv', default=PLACE_ADV, type=str, nargs="+",
                            metavar='N', help='[video relation-based adv, video-based adv, frame-based adv]')

        # ========================= Learning Configs ==========================
        self.parser.add_argument('--pretrain_source', default=False, action="store_true",
                            help='perform source-only training before DA')
        self.parser.add_argument('--epochs', default=N_EPOCH, type=int, metavar='N',  # 30
                            help='number of total epochs to run')
        self.parser.add_argument('-b', '--batch_size', default=BATCH, type=int, nargs="+",
                            # [128, 202, 128]   [32, 32, 32]    [32, 28, 64] --->tip: train80 val20 test
                            # parser.add_argument('-b', '--batch_size', default=[64, 101, 64], type=int, nargs="+",
                            metavar='N', help='mini-batch size ([source, target, testing])')
        self.parser.add_argument('--lr', '--learning_rate', default=LEARNING, type=float,  # 3e-3
                            metavar='LR', help='initial learning rate')
        self.parser.add_argument('--lr_decay', default=LRN_DECAY, # type=float,
                                 metavar='LRDecay', # 10
                            help='decay factor for learning rate')
        self.parser.add_argument('--lr_adaptive', type=str, default=LRN_ADPT, choices=['none', 'loss', 'dann'])
        self.parser.add_argument('--lr_steps', default=LRN_STEP, type=float, nargs="+",
                            metavar='LRSteps', help='epochs to decay learning rate')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        self.parser.add_argument('--weight_decay', '--wd', default=LRN_DECAY_WEIGHT, type=float,
                            metavar='W', help='weight decay (default: 1e-4)')
        self.parser.add_argument('--clip_gradient', '--gd', default=20, type=float,
                            metavar='W', help='gradient norm clipping (default: disabled)')
        self.parser.add_argument('--no_partialbn', '--npb', default=True, action="store_true")
        self.parser.add_argument('--copy_list', default=['N', 'N'], type=str, nargs="+",
                            metavar='N',
                            help='duplicate data in case the dataset is relatively small ([copy source list, copy target list])')

        # ========================= Monitor Configs ==========================
        self.parser.add_argument('--print_freq', '-pf', default=20, type=int,  # 50
                            metavar='N', help='frequency for printing to text files (default: 10)')
        self.parser.add_argument('--show_freq', '-sf', default=20, type=int,  # 50
                            metavar='N', help='frequency for showing on the screen (default: 10)')
        self.parser.add_argument('--eval_freq', '-ef', default=1, type=int,  # 5
                            metavar='N', help='evaluation frequency (default: 5)')
        self.parser.add_argument('--verbose', default=False, action="store_true")  # False

        # ========================= Runtime Configs ==========================
        self.parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',  # aumentare workers
                            # parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        self.parser.add_argument('--resume', default=RES, type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        self.parser.add_argument('--resume_hp', default=RES, action="store_true",
                            help='whether to use the saved hyper-parameters')
        self.parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='evaluate model on validation set')
        self.parser.add_argument('--exp_path', type=str,
                            default=epic_path + "model/action-model/",
                            help='full path of the experiment folder')
        # parser.add_argument('--gpus', nargs='+', type=int, default=None)
        self.parser.add_argument('--gpus', nargs='+', type=int, default=1)
        self.parser.add_argument('--flow_prefix', default="", type=str)
        self.parser.add_argument('--save_model', default=False, action="store_true")
        self.parser.add_argument('--save_best_log_val', default="best_val.log", type=str)
        self.parser.add_argument('--save_best_log_test', default="best_test.log", type=str)
        self.parser.add_argument('--save_attention', type=int, default=-1)
        self.parser.add_argument('--tensorboard', default=True, dest='tensorboard', action='store_true')

    def getParser(self):
        return self.parser
