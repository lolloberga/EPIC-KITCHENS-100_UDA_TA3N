import argparse

'''
LEGENDA:
* LIST      = LABELS     (train_val)
* DATA      = FEATURES   (prextracted_model_features)
* SOURCE    = train
* TARGET    = test
'''

ego_path = "/Users/lorenzo/University/Polito/ML and DL/EGO_Project/" # check the path for prextracted model_feature is consistent
epic_path = "/Users/lorenzo/University/Polito/ML and DL/EPIC-KITCHENS-100_UDA_TA3N/"

CURRENT_DOMAIN      = "D3"
TARGET_DOMAIN       = "D3"
FRAME_AGGREGATION   = "trn-m"
CURRENT_MODALITY    = "RGB"
USE_TARGET          = "none"
CURRENT_ARCH        = "TSM"

N_EPOCH = 50
DROP = 0.8
LEARNING = 3e-2
BATCH = [32, 32, 64]
OPTIMIZ = 'SGD'
LRN_DECAY = 10
LRN_ADPT = 'none'
LRN_STEP = list(range(5, N_EPOCH, 5))
LRN_DECAY_WEIGHT = 1e-4

RES = False

# Used only during DA
PLACE_ADV = ['N', 'N', 'N']
USE_ATTN = 'LSTA' # ['none', 'TransAttn', 'general', 'DotProduct', 'LSTA']
ADV_DA = 'none' if PLACE_ADV == ['N', 'N', 'N'] else 'RevGrad'
LOSS_ATTN = 'none' if USE_ATTN == 'none' else 'attentive_entropy'
USE_SPATIAL_FEATURES = 'Y' if USE_ATTN == 'LSTA' else 'N'


parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('--source_domain', type=str, default=CURRENT_DOMAIN)
parser.add_argument('--target_domain', type=str, default=TARGET_DOMAIN)

parser.add_argument('--num_class', type=str, default="8,8")
parser.add_argument('--modality', type=str, default=CURRENT_MODALITY)
# choices=['Audio', 'RGB', 'Flow', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus', 'ALL'])
parser.add_argument('--train_source_list', type=str,
                    default=ego_path + "train_val/" + CURRENT_DOMAIN + "_train.pkl")
parser.add_argument('--train_target_list', type=str,
                    default=ego_path + "train_val/" + CURRENT_DOMAIN + "_test.pkl")
parser.add_argument('--val_list', type=str,
                    default=ego_path + "train_val/" + CURRENT_DOMAIN + "_test.pkl")
parser.add_argument('--val_data', type=str,
                    default=ego_path + "prextracted_model_features/" + CURRENT_MODALITY + "/ek_" +
                            CURRENT_ARCH + "/" + CURRENT_DOMAIN + "-" + TARGET_DOMAIN + "_test")
parser.add_argument('--train_source_data', type=str,
                    default=ego_path + "prextracted_model_features/" + CURRENT_MODALITY + "/ek_" +
                            CURRENT_ARCH + "/" + CURRENT_DOMAIN + "-" + CURRENT_DOMAIN + "_train")
parser.add_argument('--train_target_data', type=str,
                    default=ego_path + "prextracted_model_features/" + CURRENT_MODALITY + "/ek_" +
                            CURRENT_ARCH + "/" + CURRENT_DOMAIN + "-" + TARGET_DOMAIN + "_test")

# === Spatial features (hickle files) ===
parser.add_argument('--train_source_data_spatial', type=str,
                    default=epic_path + "spatial_feat/" + CURRENT_DOMAIN + "-" + CURRENT_DOMAIN + "_train_"
                            + CURRENT_MODALITY + '_' + CURRENT_ARCH + "__spatial")
parser.add_argument('--train_target_data_spatial', type=str,
                    default=epic_path + "spatial_feat/" + CURRENT_DOMAIN + "-" + CURRENT_DOMAIN + "_test_"
                            + CURRENT_MODALITY + '_' + CURRENT_ARCH + "__spatial")
parser.add_argument('--use_spatial_features', type=str, default=USE_SPATIAL_FEATURES, choices=["N", "Y"])

# ========================= Model Configs ==========================
parser.add_argument('--train_metric', default="verb", type=str)
parser.add_argument('--dann_warmup', default=False, action="store_true")
parser.add_argument('--arch', type=str, default=CURRENT_ARCH.upper(), choices=["TBN", "I3D", "TSM"])
parser.add_argument('--pretrained', type=str, default="none")
parser.add_argument('--num_segments', type=int, default=5)
parser.add_argument('--val_segments', type=int, default=5)
parser.add_argument('--add_fc', default=1, type=int, metavar='M',
                    help='number of additional fc layers (excluding the last fc layer) (e.g. 0, 1, 2, ...)')
parser.add_argument('--fc_dim', type=int, default=512, help='dimension of added fc')
parser.add_argument('--baseline_type', type=str, default='video',
                    choices=['frame', 'video', 'tsn'])
parser.add_argument('--frame_aggregation', type=str, default=FRAME_AGGREGATION,
                    choices=['avgpool', 'rnn', 'temconv', 'trn', 'trn-m', 'none'],
                    help='aggregation of frame features (none if baseline_type is not video)')
parser.add_argument('--optimizer', type=str, default=OPTIMIZ, choices=['SGD', 'Adam'])
parser.add_argument('--use_opencv', default=False, action="store_true",
                    help='whether to use the opencv transformation')
parser.add_argument('--dropout_i', '--doi', default=0.5, type=float,
                    metavar='DOI', help='dropout ratio for frame-level feature (default: 0.5)')
parser.add_argument('--dropout_v', '--dov', default=DROP, type=float,
                    metavar='DOV', help='dropout ratio for video-level feature (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])
parser.add_argument('--weighted_class_loss', type=str, default='N', choices=['Y', 'N'])

# ------ RNN ------
parser.add_argument('--n_rnn', default=1, type=int, metavar='M',
                    help='number of RNN layers (e.g. 0, 1, 2, ...)')
parser.add_argument('--rnn_cell', type=str, default='LSTM', choices=['LSTM', 'GRU'])
parser.add_argument('--n_directions', type=int, default=1, choices=[1, 2],
                    help='(bi-) direction RNN')
parser.add_argument('--n_ts', type=int, default=5, help='number of temporal segments')

parser.add_argument('--outPool_size', type=int, default=100, help='size of output pooling for LSTA')
parser.add_argument('--mem_size', type=int, default=2048, help='size of the LSTA cell')

# ========================= DA Configs ==========================
parser.add_argument('--share_params', type=str, default='Y', choices=['Y', 'N'])
parser.add_argument('--use_target', type=str, default=USE_TARGET, choices=['none', 'Sv', 'uSv'],
                    help='the method to use target data (not use | supervised | unsupervised)')
parser.add_argument('--dis_DA', type=str, default='none', choices=['none', 'DAN', 'JAN', 'CORAL'],
                    help='discrepancy method for DA')
parser.add_argument('--adv_DA', type=str, default='RevGrad', choices=['none', 'RevGrad'],
                    help='adversarial method for DA')
parser.add_argument('--use_bn', type=str, default='none', choices=['none', 'AdaBN', 'AutoDIAL'],
                    help='normalization-based methods')
parser.add_argument('--ens_DA', type=str, default='none', choices=['none', 'MCD'], help='ensembling-based methods')
parser.add_argument('--use_attn_frame', type=str, default='none',
                    choices=['none', 'TransAttn', 'general', 'DotProduct'], help='attention-mechanism for frames only')
parser.add_argument('--use_attn', type=str, default=USE_ATTN, choices=['none', 'TransAttn', 'general', 'DotProduct', 'LSTA'],
                    help='attention-mechanism')
parser.add_argument('--n_attn', type=int, default=1, help='number of discriminators for transferable attention')
parser.add_argument('--add_loss_DA', type=str, default='none',
                    choices=['none', 'target_entropy', 'attentive_entropy'],
                    help='add more loss functions for DA')
parser.add_argument('--pred_normalize', type=str, default='N', choices=['Y', 'N'])
parser.add_argument('--alpha', default=0, type=float, metavar='M',
                    help='weighting for the discrepancy loss (use scheduler if < 0)')
parser.add_argument('--beta', default=[0.75, 0.75, 0.5], type=float, nargs="+", metavar='M',
                    help='weighting for the adversarial loss (use scheduler if < 0; [relation-beta, video-beta, frame-beta])')
parser.add_argument('--gamma', default=0.003, type=float, metavar='M',
                    help='weighting for the entropy loss')
parser.add_argument('--mu', default=0, type=float, metavar='M',
                    help='weighting for ensembling loss (e.g. discrepancy)')
parser.add_argument('--weighted_class_loss_DA', type=str, default='N', choices=['Y', 'N'])
parser.add_argument('--place_dis', default=['N', 'Y', 'N'], type=str, nargs="+",
                    metavar='N', help='where to place the discrepancy loss (length = add_fc + 2)')
parser.add_argument('--place_adv', default=['N', 'Y', 'Y'], type=str, nargs="+",
                    metavar='N', help='[video relation-based adv, video-based adv, frame-based adv]')

# ========================= Learning Configs ==========================
parser.add_argument('--pretrain_source', default=False, action="store_true",
                    help='perform source-only training before DA')
parser.add_argument('--epochs', default=N_EPOCH, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=BATCH, type=int, nargs="+",
                    # parser.add_argument('-b', '--batch_size', default=[64, 101, 64], type=int, nargs="+",
                    metavar='N', help='mini-batch size ([source, target, testing])')
parser.add_argument('--lr', '--learning_rate', default=LEARNING, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_decay', default=LRN_DECAY, type=float, metavar='LRDecay', help='decay factor for learning rate')
parser.add_argument('--lr_adaptive', type=str, default=LRN_ADPT, choices=['none', 'loss', 'dann'])
parser.add_argument('--lr_steps', default=LRN_STEP, type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=LRN_DECAY_WEIGHT, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--clip_gradient', '--gd', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=True, action="store_true")
parser.add_argument('--copy_list', default=['N', 'N'], type=str, nargs="+",
                    metavar='N',
                    help='duplicate data in case the dataset is relatively small ([copy source list, copy target list])')

# ========================= Monitor Configs ==========================
parser.add_argument('--print_freq', '-pf', default=1, type=int,
                    metavar='N', help='frequency for printing to text files (default: 10)')
parser.add_argument('--show_freq', '-sf', default=1, type=int,
                    metavar='N', help='frequency for showing on the screen (default: 10)')
parser.add_argument('--eval_freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--verbose', default=False, action="store_true")

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    # parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default=RES, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_hp', default=RES, action="store_true",
                    help='whether to use the saved hyper-parameters')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--exp_path', type=str, default=epic_path + "model/action-model/",
                    help='full path of the experiment folder')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--save_model', default=False, action="store_true")
parser.add_argument('--save_best_log_val', default="best_val.log", type=str)
parser.add_argument('--save_best_log_test', default="best_test.log", type=str)
parser.add_argument('--save_attention', type=int, default=-1)
parser.add_argument('--tensorboard', default=True, dest='tensorboard', action='store_true')
