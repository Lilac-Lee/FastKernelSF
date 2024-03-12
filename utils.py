
import time
import numpy as np
import torch
import torch.nn.functional as F
import FastGeodis

from copy import deepcopy
from collections import defaultdict
from pykdtree.kdtree import KDTree


# ANCHOR: timer!
class Timers(object):
    def __init__(self):
        self.timers = defaultdict(Timer)

    def tic(self, key):
        self.timers[key].tic()

    def toc(self, key):
        self.timers[key].toc()

    def print(self, key=None):
        if key is None:
            for k, v in self.timers.items():
                print("Average time for {:}: {:}".format(k, v.avg()))
        else:
            print("Average time for {:}: {:}".format(key, self.timers[key].avg()))

    def get_avg(self, key):
        return self.timers[key].avg()
    
    
class Timer(object):
    def __init__(self):
        self.reset()

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1

    def total(self):
        return self.total_time

    def avg(self):
        return self.total_time / float(self.calls)

    def reset(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        

# ANCHOR: early stopping strategy
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            print('loss is nan!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print('num_bad_epochs >= patience, it is {}'.format(self.num_bad_epochs))
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
 

def get_index_and_distance(pc2, pc1_deformed, k):
    pc2_numpy = pc2.detach().cpu().numpy()
    pc1_numpy = pc1_deformed.detach().cpu().numpy()
    pc1_in_pc2_indices, pc1_in_pc2_distances = get_nearest_neighbors_indices_batch(pc2_numpy, pc1_numpy, k=k)
    pc1_in_pc2_indices = torch.LongTensor(pc1_in_pc2_indices).to(pc2.device)
    pc1_in_pc2_distances = torch.from_numpy(pc1_in_pc2_distances).to(pc2.device)
    
    return pc1_in_pc2_indices, pc1_in_pc2_distances
                            

def get_nearest_neighbors_indices_batch(points_src, points_tgt, points_tgt_kdtree=None, k=1):
    ''' Returns the nearest neighbors for point sets batchwise.
    Args:
        points_src (numpy array): source points
        points_tgt (numpy array): target points
        k (int): number of nearest neighbors to return
    '''
    indices = []
    distances = []

    for i, (p1, p2) in enumerate(zip(points_src, points_tgt)):
        if points_tgt_kdtree != None:
            kdtree = points_tgt_kdtree[i]
        else:
            kdtree = KDTree(p2)
        dist, idx = kdtree.query(p1, k=k)
        indices.append(idx)
        distances.append(dist)

    return indices, distances


def chamfer_distance_kdtree(points1, points2, truncate=True, pts1_kdtree=None, pts2_kdtree=None):
    ''' KD-tree based implementation of the Chamfer distance.
    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set
        give_id (bool): whether to return the IDs of the nearest points
    '''
    # Points have size batch_size x T x 3
    batch_size = points1.size(0)
    
    # First convert points to numpy
    points1_np = points1.detach().cpu().numpy()
    points2_np = points2.detach().cpu().numpy()

    # Get list of nearest neighbors indieces
    idx_nn_12, _ = get_nearest_neighbors_indices_batch(points1_np, points2_np, pts2_kdtree)
    idx_nn_12 = torch.LongTensor(idx_nn_12).to(points1.device)
    # Expands it as batch_size x 1 x 3
    idx_nn_12_expand = idx_nn_12.view(batch_size, -1, 1).expand_as(points1)

    # Get list of nearest neighbors indieces
    idx_nn_21, _ = get_nearest_neighbors_indices_batch(points2_np, points1_np, pts1_kdtree)
    idx_nn_21 = torch.LongTensor(idx_nn_21).to(points1.device)
    # Expands it as batch_size x T x 3
    idx_nn_21_expand = idx_nn_21.view(batch_size, -1, 1).expand_as(points2)

    # Compute nearest neighbors in points2 to points in points1
    points_12 = torch.gather(points2, dim=1, index=idx_nn_12_expand)

    # Compute nearest neighbors in points1 to points in points2
    points_21 = torch.gather(points1, dim=1, index=idx_nn_21_expand)

    # Compute chamfer distance
    chamfer1 = (points1 - points_12).pow(2).sum(2)
    chamfer2 = (points2 - points_21).pow(2).sum(2)
    
    if truncate:
        # NOTE: modify the chamfer distance.
        dist_thd = 2  # we can tune this
        lengths1 = torch.full(
                    (points1.shape[0],), points1.shape[1], dtype=torch.int64, device=points1.device
                )
        lengths2 = torch.full(
                    (points2.shape[0],), points2.shape[1], dtype=torch.int64, device=points2.device
                )
        x_mask = (
            torch.arange(points1.shape[1], device=points1.device)[None] >= lengths1[:, None]
        )  # shape [N, P1]
        y_mask = (
            torch.arange(points2.shape[1], device=points2.device)[None] >= lengths2[:, None]
        )  # shape [N, P2]
        x_mask[chamfer1 >= dist_thd] = True
        y_mask[chamfer2 >= dist_thd] = True
            
        chamfer1[x_mask] = 0.0
        chamfer2[y_mask] = 0.0

    # Take sum
    chamfer = chamfer1.mean(1) + chamfer2.mean(1)

    # return chamfer.squeeze(), idx_nn_12_expand, idx_nn_21_expand
    return chamfer1, chamfer2, chamfer, idx_nn_12.view(batch_size, -1, 1), idx_nn_21.view(batch_size, -1, 1)


# ANCHOR: metrics computation, follow FlowNet3D metrics....
def scene_flow_metrics(pred, labels):
    l2_norm = torch.sqrt(torch.sum((pred - labels) ** 2, 2)).cpu()  # Absolute distance error.
    labels_norm = torch.sqrt(torch.sum(labels * labels, 2)).cpu()
    relative_err = l2_norm / (labels_norm + 1e-20)

    EPE3D = torch.mean(l2_norm).item()  # Mean absolute distance error

    # NOTE: Acc_5
    error_lt_5 = torch.BoolTensor((l2_norm < 0.05))
    relative_err_lt_5 = torch.BoolTensor((relative_err < 0.05))
    acc3d_strict = torch.mean((error_lt_5 | relative_err_lt_5).float()).item()

    # NOTE: Acc_10
    error_lt_10 = torch.BoolTensor((l2_norm < 0.1))
    relative_err_lt_10 = torch.BoolTensor((relative_err < 0.1))
    acc3d_relax = torch.mean((error_lt_10 | relative_err_lt_10).float()).item()

    # NOTE: outliers
    l2_norm_gt_3 = torch.BoolTensor(l2_norm > 0.3)
    relative_err_gt_10 = torch.BoolTensor(relative_err > 0.1)
    outlier = torch.mean((l2_norm_gt_3 | relative_err_gt_10).float()).item()

    # NOTE: angle error
    unit_label = labels / labels.norm(dim=2, keepdim=True)
    unit_pred = pred / pred.norm(dim=2, keepdim=True)
    eps = 1e-7
    dot_product = (unit_label * unit_pred).sum(2).clamp(min=-1+eps, max=1-eps)
    dot_product[dot_product != dot_product] = 0  # Remove NaNs
    angle_error = torch.acos(dot_product).mean().item()

    return EPE3D, acc3d_strict, acc3d_relax, outlier, angle_error


class DT:
    # Calculate the distance transform efficiently using tensors
    def __init__(self, pc1, pc2, grid_factor, device='cuda:0', use_dt_loss=True):
        self.device = device
        self.grid_factor = grid_factor
        
        pc1_min = torch.min(pc1, 1)[0].squeeze(0)
        pc1_max = torch.max(pc1, 1)[0].squeeze(0)
        pc2_min = torch.min(pc2, 1)[0].squeeze(0)
        pc2_max = torch.max(pc2, 1)[0].squeeze(0)
        
        xmin_int, ymin_int, zmin_int = torch.floor(torch.where(pc1_min<pc2_min, pc1_min, pc2_min\
                                            ) * grid_factor-1) / grid_factor
        xmax_int, ymax_int, zmax_int = torch.ceil(torch.where(pc1_max>pc2_max, pc1_max, pc2_max\
                                            )* grid_factor+1) / grid_factor
            
        sample_x = ((xmax_int - xmin_int) * grid_factor).ceil().int() + 2
        sample_y = ((ymax_int - ymin_int) * grid_factor).ceil().int() + 2
        sample_z = ((zmax_int - zmin_int) * grid_factor).ceil().int() + 2
        
        self.Vx = torch.linspace(0, sample_x, sample_x+1, device=self.device)[:-1] / grid_factor + xmin_int
        self.Vy = torch.linspace(0, sample_y, sample_y+1, device=self.device)[:-1] / grid_factor + ymin_int
        self.Vz = torch.linspace(0, sample_z, sample_z+1, device=self.device)[:-1] / grid_factor + zmin_int
        
        # NOTE: build a binary image first, with 0-value occuppied points, then use opencv function
        grid_x, grid_y, grid_z = torch.meshgrid(self.Vx, self.Vy, self.Vz, indexing="ij")
        self.grid = torch.stack([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1), grid_z.unsqueeze(-1)], -1).float().squeeze()
        
        if use_dt_loss:
            H, W, D, _ = self.grid.size()
            pts_mask = torch.ones(H, W, D, device=device)
            self.pts_sample_idx_x = ((pc2[...,0:1] - self.Vx[0]) * self.grid_factor).round()
            self.pts_sample_idx_y = ((pc2[...,1:2] - self.Vy[0]) * self.grid_factor).round()
            self.pts_sample_idx_z = ((pc2[...,2:3] - self.Vz[0]) * self.grid_factor).round()
            pts_mask[self.pts_sample_idx_x.long(), self.pts_sample_idx_y.long(), self.pts_sample_idx_z.long()] = 0.
            
            iterations = 1
            image_pts = torch.zeros(H, W, D, device=device).unsqueeze(0).unsqueeze(0)
            pts_mask = pts_mask.unsqueeze(0).unsqueeze(0)
            self.D = FastGeodis.generalised_geodesic3d(
                image_pts, pts_mask, [1./self.grid_factor, 1./self.grid_factor, 1./self.grid_factor], 1e10, 0.0, iterations
            ).squeeze()
        else:
            self.D = deepcopy(self.grid)
            
    def torch_bilinear_distance(self, Y):
        H, W, D = self.D.size()
        target = self.D[None, None, ...]
        
        sample_x = ((Y[:,0:1] - self.Vx[0]) * self.grid_factor).clip(0, H-1)
        sample_y = ((Y[:,1:2] - self.Vy[0]) * self.grid_factor).clip(0, W-1)
        sample_z = ((Y[:,2:3] - self.Vz[0]) * self.grid_factor).clip(0, D-1)
        
        sample = torch.cat([sample_x, sample_y, sample_z], -1)
        
        # NOTE: normalize samples to [-1, 1]
        sample = 2 * sample
        sample[...,0] = sample[...,0] / (H-1)
        sample[...,1] = sample[...,1] / (W-1)
        sample[...,2] = sample[...,2] / (D-1)
        sample = sample -1
        sample_ = torch.cat([sample[...,2:3], sample[...,1:2], sample[...,0:1]], -1)
        
        dist = F.grid_sample(target, sample_.view(1,-1,1,1,3), mode="bilinear", align_corners=True).view(-1)
        
        return dist

