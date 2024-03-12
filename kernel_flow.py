import os
import argparse
import logging
import time
import math
import pandas as pd
import torch

from torch.utils.data import DataLoader

import config
from data import ArgoverseSceneFlowDataset, WaymoOpenFlowDataset
from model import Neural_Prior, encoding_func_3D
from utils import Timers, DT, chamfer_distance_kdtree, scene_flow_metrics, EarlyStopping
from visualization import show_flows
from pykdtree.kdtree import KDTree


def init_params(param_shape, init_method='', init_scaling=1.0, device='cuda:0', requires_grad=True):
    if init_method == 'same_as_linear':
        stdv = 1. / math.sqrt(param_shape[1]*param_shape[2])
        param = torch.distributions.Uniform(-stdv, stdv).sample(param_shape)
    
    param = param.to(device)
    if requires_grad:
        param.requires_grad = True
    
    return param


def main(options):
    if 'argoverse' in options.dataset_name:
        data_loader = DataLoader(ArgoverseSceneFlowDataset(options=options, partition=options.partition), \
                                batch_size=options.batch_size, shuffle=False, drop_last=False, num_workers=12)
    elif 'waymo' in options.dataset_name:
        data_loader = DataLoader(WaymoOpenFlowDataset(options=options, partition=options.partition), \
                                batch_size=options.batch_size, shuffle=False, drop_last=False, num_workers=12)
    
    outputs = []
    if options.time:
        timers = Timers()
        timers.tic("total_time")

    for i, data in enumerate(data_loader):
        if options.earlystopping:	
            early_stopping = EarlyStopping(patience=options.early_patience, min_delta=options.early_min_delta)	
            
        pre_compute_st = time.time()
        solver_time = 0.
        
        pc1, pc2, flow = data
        
        pc1 = pc1.to(options.device).contiguous()
        pc2 = pc2.to(options.device).contiguous()
        flow = flow.to(options.device).contiguous()
        
        # ANCHOR: loss preprocessing -- do not need since we have GT flow
        if options.use_dt_loss:
            dt = DT(pc1, pc2, grid_factor=options.dt_grid_factor, device=options.device, use_dt_loss=True)
        elif options.use_chamfer:
            pc2_kdtree = []
            for k in range(pc2.shape[0]):
                pc2_kdtree.append(KDTree(pc2[k].detach().cpu().numpy()))
        
        # ANCHOR: kernel function
        if options.kernel_grid:   # K(p1,p*)
            # ANCHOR: for complex encoding grid computation, similar to building a DT map
            complex_grid = DT(pc1, pc2, grid_factor=options.grid_factor, device=options.device, use_dt_loss=False)
            grid_pts = complex_grid.grid
            pc2_ = grid_pts.reshape(-1, pc1.shape[-1]).unsqueeze(0)
        else:
            pc2_ = pc2.clone()
            
        if options.model == 'none':
            # NOTE: for point-based kernel
            feats1_loc = pc1.clone()
            feats2_loc = pc2_.clone()
        elif options.model == 'pe':	
            pe3d = encoding_func_3D(options.pe_type, param=(options.pe_sigma, options.pe_dim), device=options.device, dim_x=3)
            feats1_loc = pe3d(pc1)
            feats2_loc = pe3d(pc2_)
        
        # NOTE: pc1 -- observation; kernel grid -- known points; therefore, alpha should have the same size as kernel grid.
        feats1_gram = torch.linalg.norm(feats1_loc, dim=-1, keepdim=True) ** 2
        feats2_gram = torch.linalg.norm(feats2_loc, dim=-1, keepdim=True) ** 2
        feats1_dot_feats2 = torch.einsum('ijk,ilk->ijl', feats1_loc, feats2_loc)
        rbf = feats1_gram + feats2_gram.permute(0,2,1) - 2 * feats1_dot_feats2
        
        if options.kernel_type == 'gaussian':
            rbf = torch.exp(-1./(2*options.log_sigma) * rbf)   # BxNxK
            
        # ANCHOR: set optimization parameters, and begin optimization
        alpha = init_params((options.batch_size, rbf.shape[2], 3), init_method=options.alpha_init_method, init_scaling=options.alpha_init_scaling, device=options.device, requires_grad=True)
        param = [{'params': alpha, 'lr': options.alpha_lr, 'weight_decay': options.weight_decay}]
        optimizer = torch.optim.Adam(param)
        
        # ANCHOR: initialize best metrics
        best_loss = 1e10
        best_flow = None
        best_epe3d = 1.
        best_acc3d_strict = 0.
        best_acc3d_relax = 0.
        best_angle_error = 1.
        best_outliers = 1.
        best_epoch = 0
        
        pre_compute_time = time.time() - pre_compute_st
        solver_time = solver_time + pre_compute_time
        
        for epoch in range(options.iters):
            iter_time_init = time.time()
            
            flow_pred = alpha.transpose(1,2).bmm(rbf.transpose(1,2)).transpose(1,2)
            pc1_deformed = pc1 + flow_pred
            
            if options.use_dt_loss:
                loss_corr = dt.torch_bilinear_distance(pc1_deformed.squeeze(0)).mean()
            elif options.use_chamfer:
                _, _, loss_corr, _, _ = chamfer_distance_kdtree(pc2, pc1_deformed, truncate=options.truncate_cd, \
                                            pts1_kdtree=pc2_kdtree)
               
            # NOTE: add TV regularizer?
            if options.reg_name == 'l1':
                reg_scaled = options.reg_scaling * alpha.abs().mean()
            elif options.reg_name == 'l2':
                reg_scaled = options.reg_scaling * (alpha ** 2).mean()
            
            if options.reg_name != 'none':
                loss = loss_corr + reg_scaled
            else:
                loss = loss_corr
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iter_time = time.time() - iter_time_init
            solver_time = solver_time + iter_time
            
            flow_pred_final = pc1_deformed - pc1
            flow_metrics = flow.detach().clone()
            
            epe3d, acc3d_strict, acc3d_relax, outlier, angle_error = scene_flow_metrics(\
                                                    flow_pred_final, flow_metrics)
            
            # ANCHOR: get best metrics
            if loss <= best_loss:
                best_loss = loss.item()
                best_flow = flow_pred_final
                best_epe3d = epe3d
                best_acc3d_strict = acc3d_strict
                best_acc3d_relax = acc3d_relax
                best_angle_error = angle_error
                best_outliers = outlier
                best_epoch = epoch
            
            if options.earlystopping:
                if early_stopping.step(loss):
                    break
                
        info_dict = {
            'final_flow': best_flow,
            'loss': best_loss,
            'EPE3D': best_epe3d,
            'acc3d_strict': best_acc3d_strict * 100,
            'acc3d_relax': best_acc3d_relax * 100,
            'angle_error': best_angle_error,
            'outlier': best_outliers * 100,
            'epoch': best_epoch,
            'solver_time': solver_time,
        }
        
        outputs.append(dict(list(info_dict.items())[1:]))
        
        if options.visualize:
            idx = 0
            show_flows(pc1[idx], pc2[idx], flow_metrics[idx])
            show_flows(pc1[idx], pc2[idx], flow_pred_final[idx])
            
        if options.time:
            timers.toc("total_time")
            logging.info(timers.print())
        
        print('Optimizing at example {}'.format(i))
        print(dict(list(info_dict.items())[1:]))
        
    df = pd.DataFrame(outputs)
    df.loc['mean'] = df.mean()
    logging.info(df.mean())
    print('Finished Optimizing')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Neural Kernel Scene Flow.")
    config.add_config(parser)
    options = parser.parse_args()
    
    os.makedirs(options.exp_dir_path, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.FileHandler(filename=f"{options.exp_dir_path}/run.log"), logging.StreamHandler()])
    logging.info(options)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    torch.random.manual_seed(1234)
    
    main(options)
