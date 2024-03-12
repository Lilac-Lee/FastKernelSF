import argparse
import os

import numpy as np
import torch


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def init_dirs(options):
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    if not os.path.exists(f"checkpoints/{options.exp_name}"):
        os.makedirs(f"checkpoints/{options.exp_name}")
    if not os.path.exists(f"checkpoints/{options.exp_name}/models"):
        os.makedirs(f"checkpoints/{options.exp_name}/models")
    os.system(f"cp main.py checkpoints/{options.exp_name}/main.py.backup")
    os.system(f"cp model.py checkpoints/{options.exp_name}/model.py.backup")
    os.system(f"cp data.py checkpoints/{options.exp_name}/data.py.backup")


def set_deterministic_seeds(options):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed_all(options.seed)
    np.random.seed(options.seed)


def add_config(parser):
    parser.add_argument('--exp_dir_path', type=str, default='./logs/fast_kernel_scene_flow', metavar='N', help='experiment path.')
    parser.add_argument('--num_points', type=int, default=8192, help='Point number [default: 2048].')
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size', help='Batch size.')
    parser.add_argument('--device', default='cuda:0', type=str, help='device: cpu? cuda?')
    parser.add_argument('--seed', type=int, default=1234, metavar='S', help='Random seed (default: 1234).')
    parser.add_argument('--dataset_name', type=str, default='argoverse', metavar='N', help='Dataset to use.')
    parser.add_argument('--data_path', type=str, default='./dataset', metavar='N', help='path to data.')
    parser.add_argument('--partition', type=str, default='val', metavar='p', help='Model to use.')
    parser.add_argument('--visualize', action='store_true', default=False, help='Show visuals.')
    parser.add_argument('--time', dest='time', action='store_true', default=True, help='Count the execution time of each step.')
    parser.add_argument('--iters', type=int, default=10, help='number of total iterations to solve.')
    parser.add_argument('--earlystopping', action='store_true', default=False, help='whether to use early stopping or not.')
    parser.add_argument('--early_patience', type=int, default=100, help='patience in early stopping.')
    parser.add_argument('--early_min_delta', type=float, default=0.0001, help='the minimum delta of early stopping.')
    parser.add_argument('--compute_metrics', action='store_true', default=False, help='whether to compute metrics or not.')
    parser.add_argument('--weight_decay', type=float, default=0., metavar='N', help='Weight decay.')
    parser.add_argument('--use_all_points', action='store_true', default=False, help='use all the points or not.')

    # ANCHOR: loss functions
    parser.add_argument('--use_chamfer', action='store_true', default=False, help='whether to use Chamfer loss.')
    parser.add_argument('--truncate_cd', action='store_true', default=False, help='whether to truncate the Chamfer loss or not.')
    parser.add_argument('--use_dt_loss', action='store_true', default=False, help='whether to use DT loss.')
    
    # ANCHOR: for regularizer
    parser.add_argument('--reg_name', type=str, default='none', choices=['none', 'l1', 'l2'], help='which regularizer to use.')
    parser.add_argument('--reg_scaling', type=float, default=1., help='scaling factor for regularizer.')
    parser.add_argument('--epsilon', type=float, default=1e-5, help='epsilon to prevent divide by zeros in regularizer.')
    
    # ANCHOR: for kernel learning
    parser.add_argument('--model', type=str, default='none', metavar='N', help='Model to use, none or pe.')
    parser.add_argument('--kernel_grid', action='store_true', default=False, help='whether to use grid to compute K(p1,p*) or not.')
    parser.add_argument('--pe_type', type=str, default='RFF', help='positional encoding type.')
    parser.add_argument('--pe_sigma', type=float, default=1., help='scale for pe.')
    parser.add_argument('--pe_dim', type=int, default=256, help='dimension for pe.')
    parser.add_argument('--pe_kernel', action='store_true', default=False, help='whether to use positional encoding-based kernel or not.')
    parser.add_argument('--log_sigma', type=float, default=22.5, help='scaling factor for exp.')
    parser.add_argument('--kernel_type', type=str, default='gaussian', help='which kernel to use.')
    parser.add_argument('--grid_factor', type=float, default=2., help='grid size.')
    parser.add_argument('--dt_grid_factor', type=float, default=2., help='grid size for DT.')
    parser.add_argument('--alpha_lr', type=float, default=0.001, help='learning rate for alpha optimization.')
    parser.add_argument('--alpha_init_method', type=str, default='same_as_linear', help='which initialization for weights to use for alpha.')
    parser.add_argument('--alpha_init_scaling', type=float, default=1., help='scaling factor for weight initialization for alpha.')
    