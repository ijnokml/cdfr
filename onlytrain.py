# import pytorch
# python -m torch.distributed.launch --nproc_per_node 2 onlytrain.py
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

# Other
import os
import time
import numpy as np
import argparse

from build_data import build_loader
import get_config

from train_eval import train

# #
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default = -1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()
CFG_PATH = 'configs/swin.yaml'
config = get_config.get_cfg(CFG_PATH)
config.defrost()
if args.local_rank == -1:
    args.local_rank = 'cpu'
config.SYSTEM.LOCAL_RANK=args.local_rank
config.freeze()
# 类别
# 0 - unchanged (False)
# 1 - changed
#############################################
if config.SYSTEM.LOCAL_RANK != 'cpu':
    torch.cuda.set_device(config.SYSTEM.LOCAL_RANK) 
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()#用于同步进程状态

    cudnn.benchmark = True

    seed = config.SYSTEM.SEED + dist.get_rank()
else:
    seed = config.SYSTEM.SEED

torch.manual_seed(seed)
np.random.seed(seed)

if config.SYSTEM.LOCAL_RANK != 'cpu':
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()
    #############################################

dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config=config)

# load model
t_start = time.time()
train(config=config,data_loader_train=data_loader_train,val_loader=data_loader_val)
t_end = time.time()
print(f'Elapsed time:{t_end - t_start}')