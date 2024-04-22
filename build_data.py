import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset
from skimage import io
from os.path import join

from augmentations import *
from preprocessors import Normalize


def build_loader(config):

    data_transform_train = (Compose(Choose(HorizontalFlip(), VerticalFlip(), Rotate('90'), Rotate('180'),
                                           Rotate('270'),  Shift(),),),#Scale([0.8, 3.0]),#FIXME
                            Normalize(0.0, 255.0), None)
    data_transform_eval = (None, Normalize(0.0, 255.0), None)
    if config.AUG:
        pass
    else:
        data_transform_train = data_transform_eval
    print('Dataset '+config.DATA.DATASET+' is loaded.')

    dataset_train = globals()[config.DATA.DATASET](
        path=config.DATA.DATA_PATH, mode='train', transform=data_transform_train)

    dataset_val = globals()[config.DATA.DATASET](
        path=config.DATA.DATA_PATH, mode='val', transform=data_transform_eval)

    # test_dataset = CCD_Dataset(path=PATH_TO_DATASET, mode='test',
    #                        transform=data_transform_eval)
    if config.SYSTEM.LOCAL_RANK != 'cpu':
        print(
            f"local rank {config.SYSTEM.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
        print(
            f"local rank {config.SYSTEM.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)

        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        #TODO
    else:
        sampler_train = None
        sampler_val = None


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.SYSTEM.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.SYSTEM.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )



    return dataset_train, dataset_val, data_loader_train, data_loader_val


class CCD(Dataset):
    def __init__(self, path, mode='train', transform=None):
        self.transform = transform
        self.transform = list(self.transform)
        self.transform += [None]*(3-len(self.transform)) 
        self.path = path
        self.mode = mode
        self.I1_list, self.I2_list, self.cm_list, self.namelist = self.read_file_path()
        self.len = len(self.cm_list)

    def read_file_path(self):
        if self.mode == 'train':
            fname = 'train.txt'
        elif self.mode == 'test':
            fname = 'test.txt'
        elif self.mode == 'val':
            fname = 'val.txt'
        else:
            raise RuntimeError('using undifined mode, ERROR')
        I1_list = []
        I2_list = []
        cm_list = []
        file_set = open(self.path + fname, 'r')
        name_list = file_set.readlines()
        for i in name_list:
            I1_list.append(join(self.path, self.mode, 'A', i[:-1]))
            I2_list.append(join(self.path, self.mode, 'B', i[:-1]))
            cm_list.append(join(self.path, self.mode, 'OUT', i[:-1]))
        return I1_list, I2_list, cm_list, name_list

    def fetch_img(self, impath):
        return io.imread(impath)

    def fetch_cm(self, cmpath):
        # imgs are NOT bool, and the pixel values range from 0 to 255.
        return (io.imread(cmpath) > 127).astype(np.bool)

    def to_tensor(self, num_array):
        if any(s < 0 for s in num_array.strides):
            num_array = np.ascontiguousarray(num_array)
        if num_array.ndim == 3:
            return torch.from_numpy(np.transpose(num_array, (2, 0, 1)))
        # if num_array.ndim == 1:

        else:
            return torch.from_numpy(num_array)

    def preprocess(self, I1, I2, cm):
        if self.transform[0] is not None:
            # Applied to all
            I1, I2, cm = self.transform[0](I1, I2, cm)
        if self.transform[1] is not None:
            # Solely for images
            I1, I2 = self.transform[1](I1, I2)
        if self.transform[2] is not None:
            # Solely for labels
            cm = self.transform[2](cm)
        torch_data1 = self.to_tensor(I1).float(),
        torch_data2 = self.to_tensor(I2).float(),
        torch_datac = self.to_tensor(cm)
        # .float()
        torch_data = torch_data1, torch_data2, torch_datac
        return torch_data

    def fetch_and_preprocess(self, index):
        I1 = self.fetch_img(self.I1_list[index])
        I2 = self.fetch_img(self.I2_list[index])
        cm = self.fetch_cm(self.cm_list[index])  # bool
        I1, I2, cm = self.preprocess(I1, I2, cm)
        return I1, I2, cm

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if index >= len(self) or index < 0:
            print(index)
            raise IndexError

        item = self.fetch_and_preprocess(index)

        return item 


class LEVIR256(Dataset):
    def __init__(self, path, mode='train', transform=None):
        self.transform = transform
        self.transform = list(self.transform)
        self.transform += [None]*(3-len(self.transform))  
        self.path = path
        self.mode = mode
        self.I1_list, self.I2_list, self.cm_list, self.namelist = self.read_file_path()
        self.len = len(self.cm_list)

    def read_file_path(self):
        if self.mode == 'train':
            fname = 'train.txt'
        elif self.mode == 'test':
            fname = 'test.txt'
        elif self.mode == 'val':
            fname = 'val.txt'
        else:
            raise RuntimeError('using undifined mode, ERROR')
        I1_list = []
        I2_list = []
        cm_list = []
        file_set = open(self.path + fname, 'r')
        name_list = file_set.readlines()
        for i in name_list:
            I1_list.append(join(self.path, 'A', i[:-1]))
            I2_list.append(join(self.path, 'B', i[:-1]))
            cm_list.append(join(self.path, 'label', i[:-1]))
        return I1_list, I2_list, cm_list, name_list

    def fetch_img(self, impath):
        return io.imread(impath)

    def fetch_cm(self, cmpath):
        # imgs are NOT bool, and the pixel values range from 0 to 255.
        return (io.imread(cmpath) > 127).astype(np.bool)

    def to_tensor(self, num_array):
        if any(s < 0 for s in num_array.strides):
            num_array = np.ascontiguousarray(num_array)
        if num_array.ndim == 3:
            return torch.from_numpy(np.transpose(num_array, (2, 0, 1)))
        # if num_array.ndim == 1:

        else:
            return torch.from_numpy(num_array)

    def preprocess(self, I1, I2, cm):
        if self.transform[0] is not None:
            # Applied to all
            I1, I2, cm = self.transform[0](I1, I2, cm)
        if self.transform[1] is not None:
            # Solely for images
            I1, I2 = self.transform[1](I1, I2)
        if self.transform[2] is not None:
            # Solely for labels
            cm = self.transform[2](cm)
        torch_data1 = self.to_tensor(I1).float(),
        torch_data2 = self.to_tensor(I2).float(),
        torch_datac = self.to_tensor(cm)
        # .float()
        torch_data = torch_data1, torch_data2, torch_datac
        return torch_data

    def fetch_and_preprocess(self, index):
        I1 = self.fetch_img(self.I1_list[index])
        I2 = self.fetch_img(self.I2_list[index])
        cm = self.fetch_cm(self.cm_list[index])  # bool
        I1, I2, cm = self.preprocess(I1, I2, cm)
        return I1, I2, cm

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if index >= len(self) or index < 0:
            print(index)
            raise IndexError

        item = self.fetch_and_preprocess(index)

        return item 

class DSIFN256(Dataset):
    def __init__(self, path, mode='train', transform=None):
        self.transform = transform
        self.transform = list(self.transform)
        self.transform += [None]*(3-len(self.transform))  
        self.path = path
        self.mode = mode
        self.I1_list, self.I2_list, self.cm_list, self.namelist = self.read_file_path()
        self.len = len(self.cm_list)

    def read_file_path(self):
        if self.mode == 'train':
            fname = 'train.txt'
        elif self.mode == 'test':
            fname = 'test.txt'
        elif self.mode == 'val':
            fname = 'val.txt'
        else:
            raise RuntimeError('using undifined mode, ERROR')
        I1_list = []
        I2_list = []
        cm_list = []
        file_set = open(self.path + fname, 'r')
        name_list = file_set.readlines()
        for i in name_list:
            I1_list.append(join(self.path, 'A', i[:-1]))
            I2_list.append(join(self.path, 'B', i[:-1]))
            cm_list.append(join(self.path, 'label', i[:-1]))
        return I1_list, I2_list, cm_list, name_list

    def fetch_img(self, impath):
        return io.imread(impath)

    def fetch_cm(self, cmpath):
        # imgs are NOT bool, and the pixel values range from 0 to 255.
        return (io.imread(cmpath) > 127).astype(np.bool)

    def to_tensor(self, num_array):
        if any(s < 0 for s in num_array.strides):
            num_array = np.ascontiguousarray(num_array)
        if num_array.ndim == 3:
            return torch.from_numpy(np.transpose(num_array, (2, 0, 1)))
        # if num_array.ndim == 1:

        else:
            return torch.from_numpy(num_array)

    def preprocess(self, I1, I2, cm):
        if self.transform[0] is not None:
            # Applied to all
            I1, I2, cm = self.transform[0](I1, I2, cm)
        if self.transform[1] is not None:
            # Solely for images
            I1, I2 = self.transform[1](I1, I2)
        if self.transform[2] is not None:
            # Solely for labels
            cm = self.transform[2](cm)
        torch_data1 = self.to_tensor(I1).float(),
        torch_data2 = self.to_tensor(I2).float(),
        torch_datac = self.to_tensor(cm)
        # .float()
        torch_data = torch_data1, torch_data2, torch_datac
        return torch_data

    def fetch_and_preprocess(self, index):
        I1 = self.fetch_img(self.I1_list[index])
        I2 = self.fetch_img(self.I2_list[index])
        cm = self.fetch_cm(self.cm_list[index])  # bool
        I1, I2, cm = self.preprocess(I1, I2, cm)
        return I1, I2, cm

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if index >= len(self) or index < 0:
            print(index)
            raise IndexError

        item = self.fetch_and_preprocess(index)

        return item 

