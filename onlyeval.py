import torch
import torch.backends.cudnn as cudnn

# Other
import time
import os
import numpy as np


from build_data import build_loader
import get_config

from evalfunc import test

if not os.path.exists('./savepic'):
    
    os.makedirs('./savepic')
# #
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
CFG_PATH = 'swin_upernet.yaml'
config = get_config.get_cfg(CFG_PATH)

#############################################
seed = config.SYSTEM.SEED
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.benchmark = True
#############################################

dataset_test, dataset_val, data_loader_test, data_loader_val = build_loader(config=config)

t_start = time.time()
torch.cuda.synchronize()
test(config=config,data_loader_test=data_loader_test, val_loader=data_loader_val)
torch.cuda.synchronize()
t_end = time.time()
print(f'Elapsed time:{t_end - t_start}')