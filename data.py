import os, glob
import numpy as np
from torch.utils.data import Dataset


class ArgoverseSceneFlowDataset(Dataset):
    def __init__(self, options, partition="val", width=1):
        self.options = options
        self.partition = partition
        self.width = width
        self.num_points = options.num_points
        
        if self.partition == "val":
            self.datapath = sorted(glob.glob(os.path.join(options.data_path, options.dataset_name, 'val', '*/*.npz')))
        
        print('number of data is {}'.format(len(self.datapath)))
            
    def __len__(self):
        return len(self.datapath)
    
    def __getitem__(self, index):
        filename = self.datapath[index]
        
        with open(filename, 'rb') as fp:
            data = np.load(fp)
            pc1 = data['pc1']
            pc2 = data['pc2']
            flow = data['flow']
            
        if not self.options.use_all_points:
            rand_idx = np.random.choice(pc1.shape[0],self.options.num_points)
            pc1 = pc1[rand_idx]
            pc2 = pc2[rand_idx]
            flow = flow[rand_idx]
        
        return pc1, pc2, flow
    
    
class WaymoOpenFlowDataset(Dataset):
    def __init__(self, options, partition="val", width=1):
        self.options = options
        self.partition = partition
        self.width = width
        self.num_points = options.num_points
        
        if self.partition == "val":
            self.datapath = sorted(glob.glob(os.path.join(options.data_path, options.dataset_name, '*/*.npz')))
            
        print('number of data is {}'.format(len(self.datapath)))
            
    def __len__(self):
        return len(self.datapath)
    
    def __getitem__(self, index):
        filename = self.datapath[index]
        
        with open(filename, 'rb') as fp:
            data = np.load(fp)
            pc1 = data['pc1']
            pc2 = data['pc2']
            flow = data['flow']
            
        if not self.options.use_all_points:
            rand_idx = np.random.choice(pc1.shape[0],self.options.num_points)
            pc1 = pc1[rand_idx]
            pc2 = pc2[rand_idx]
            flow = flow[rand_idx]
        
        return pc1, pc2, flow