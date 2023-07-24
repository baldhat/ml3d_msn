import sys
import open3d as o3d
from model import *
from utils import *
import argparse
import random
import numpy as np
import torch
import os
import visdom
sys.path.append("./emd/")
import emd_module as emd
from model_utils import calc_dcd

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = './trained_model/network.pth',  help='optional reload model path')
parser.add_argument('--num_points', type=int, default = 8192,  help='number of points')
parser.add_argument('--n_primitives', type=int, default = 16,  help='number of primitives in the atlas')
parser.add_argument('--env', type=str, default ="MSN_VAL"   ,  help='visdom environment') 
parser.add_argument('--loss', type=int, default = 1  ,  help='1 = EMD, 2 = DCD,  3=CD') 
parser.add_argument('--n_lambda', type=int, default = 200   ,  help='n lambda for dcd loss') 
parser.add_argument('--t_alpha', type=float, default = 0.5   ,  help='t_alpha for dcd loss') 



opt = parser.parse_args()
print (opt)

network = MSN(num_points = opt.num_points, n_primitives = opt.n_primitives) 
network.cuda()
network.apply(weights_init)

vis = visdom.Visdom(port = 8097, env=opt.env) # set your port

if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print("Previous weight loaded ")

network.eval()
with open(os.path.join('./data/val.list')) as file:
    model_list = [line.strip().replace('/', '_') for line in file]

partial_dir = "./data/val/"
gt_dir = "./data/complete/" 
vis = visdom.Visdom(port = 8097, env=opt.env) # set your port

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

EMD = emd.emdModule()

labels_generated_points = torch.Tensor(range(1, (opt.n_primitives+1)*(opt.num_points//opt.n_primitives)+1)).view(opt.num_points//opt.n_primitives,(opt.n_primitives+1)).transpose(0,1)
labels_generated_points = (labels_generated_points)%(opt.n_primitives+1)
labels_generated_points = labels_generated_points.contiguous().view(-1)

with torch.no_grad():
    for i, model in enumerate(model_list):
        print(model)
        partial = torch.zeros((50, 5000, 3), device='cuda')
        gt = torch.zeros((50, opt.num_points, 3), device='cuda')
        for j in range(50):
            pcd = o3d.io.read_point_cloud(os.path.join(partial_dir, model + '_' + str(j) + '_denoised.pcd'))
            partial[j, :, :] = torch.from_numpy(resample_pcd(np.array(pcd.points), 5000))
            pcd = o3d.io.read_point_cloud(os.path.join(gt_dir, model + '.pcd'))
            gt[j, :, :] = torch.from_numpy(resample_pcd(np.array(pcd.points), opt.num_points))

        output1, output2, expansion_penalty = network(partial.transpose(2,1).contiguous())
        if opt.loss == 1:
            emd_dcd1, _, _ = calc_dcd(output1, gt, alpha=opt.t_alpha, n_lambda=opt.n_lambda)
            emd_dcd1 = emd_dcd1.mean()
            emd_dcd2, _, _ = calc_dcd(output2, gt, alpha=opt.t_alpha, n_lambda=opt.n_lambda)
            emd_dcd2 = emd_dcd2.mean()
        elif opt.loss == 2 :
            dist, _ = EMD(output1, gt, 0.002, 10000)
            emd_dcd1 = torch.sqrt(dist).mean()
            dist, _ = EMD(output2, gt, 0.002, 10000)
            emd_dcd2 = torch.sqrt(dist).mean()
        else:
            raise NotImplementedError("Not implemented yet")
        idx = random.randint(0, 49)
        vis.scatter(X = gt[idx].data.cpu(), win = 'GT',
                    opts = dict(title = model, markersize = 2))
        vis.scatter(X = partial[idx].data.cpu(), win = 'INPUT',
                    opts = dict(title = model, markersize = 2))
        vis.scatter(X = output1[idx].data.cpu(),
                    Y = labels_generated_points[0:output1.size(1)],
                    win = 'COARSE',
                    opts = dict(title = model, markersize=2))
        vis.scatter(X = output2[idx].data.cpu(),
                    win = 'OUTPUT',
                    opts = dict(title = model, markersize=2))
        # print(opt.env + ' val [%d/%d]  dcd1: %f dcd2: %f expansion_penalty: %f' %(i + 1, len(model_list), emd1.item(), emd2.item(), expansion_penalty.mean().item()))
        if opt.loss ==1:
            print(opt.env + '  val [%d/%d]  emd1: %f emd2: %f expansion_penalty: %f' %(i + 1, len(model_list), emd_dcd1.item(), emd_dcd2.item(), expansion_penalty.mean().item()))
        elif opt.loss ==2:
            print(opt.env + '  val [%d/%d]  dcd1: %f dcd2: %f expansion_penalty: %f' %(i + 1, len(model_list), emd_dcd1.item(), emd_dcd2.item(), expansion_penalty.mean().item()))
        else:
            raise NotImplementedError("Not implemented yet")
