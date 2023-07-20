import open3d as o3d
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import sys
from dataset import *
from model import *
from utils import *
import os
import json
import time, datetime
import visdom
from time import time
sys.path.append("./emd/")
import emd_module as emd
from model_utils import calc_dcd


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = '',  help='optional reload model path')
parser.add_argument('--num_points', type=int, default = 8192,  help='number of points')
parser.add_argument('--n_primitives', type=int, default = 16,  help='number of surface elements')
parser.add_argument('--env', type=str, default ="MSN_TRAIN"   ,  help='visdom environment')
parser.add_argument('--encoder', type=int, default =1   ,  help='which feature encoder to use 1=pointnet 2=pointnet++ 3=pointconv')
parser.add_argument('--loss', type=int, default =2  ,  help='which loss to use 1=emd 2=dcd')



opt = parser.parse_args()
print (opt)

class FullModel(nn.Module):
    def __init__(self, model , dcd = False):
        super(FullModel, self).__init__()
        self.model = model
        self.EMD = emd.emdModule()
        self.dcd = dcd
        self.t_alpha = 200
        self.n_lambda = 0.5

    def forward(self, inputs, gt, eps, iters):
        output1, output2, expansion_penalty = self.model(inputs)
        gt = gt[:, :, :3] 
        
        if self.dcd:
            dcd1, _, _ = calc_dcd(output1, gt, alpha=self.t_alpha, n_lambda=self.n_lambda)
            dcd1 = dcd1.mean()
            dcd2, _, _ = calc_dcd(output2, gt, alpha=self.t_alpha, n_lambda=self.n_lambda)
            dcd2 = dcd2.mean()
            return output1, output2, dcd1, dcd2 , expansion_penalty

        else:
            dist, _ = self.EMD(output1, gt, eps, iters)
            emd1 = torch.sqrt(dist).mean(1)
            dist, _ = self.EMD(output2, gt, eps, iters)
            emd2 = torch.sqrt(dist).mean(1)    
        
        return output1, output2, emd1, emd2 , expansion_penalty

if __name__ == '__main__':
    vis = visdom.Visdom(port = 8097, env=opt.env) # set your port
    now = datetime.datetime.now()
    save_path = str(now.timestamp())
    if not os.path.exists('./log/'):
        os.mkdir('./log/')
    dir_name =  os.path.join('log', save_path)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    logname = os.path.join(dir_name, 'log.txt')
    os.system('copy ./train.py %s' % dir_name)
    os.system('copy ./dataset.py %s' % dir_name)
    os.system('copy ./model.py %s' % dir_name)

    opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    best_val_loss = 10

    # dataset = ShapeNet(train=True, npoints=opt.num_points)
    dataset = ModelNet(split="train", npoints=opt.num_points, classes=["bed", "bathtub", "toilet", "monitor", "sofa"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=int(opt.workers))
    # dataset_test = ShapeNet(train=False, npoints=opt.num_points)
    dataset_val = ModelNet(split="val", npoints=opt.num_points, classes=["bed", "bathtub", "toilet", "monitor", "sofa"])
    dataloader_test = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize,
                                                  shuffle=False, num_workers=int(opt.workers))

    len_dataset = len(dataset)
    print("Train Set Size: ", len_dataset)
    print("Test Set Size: ", len(dataset_val))

    network = MSN(num_points = opt.num_points, n_primitives = opt.n_primitives, feature_extractor= opt.encoder)
    network = torch.nn.DataParallel(FullModel(network))
    network.cuda()
    network.module.model.apply(weights_init) #initialization of the weight

    if opt.model != '':
        network.module.model.load_state_dict(torch.load(opt.model))
        print("Previous weight loaded ")

    lrate = 0.001 #learning rate
    optimizer = optim.Adam(network.module.model.parameters(), lr = lrate)

    train_loss = AverageValueMeter()
    val_loss = AverageValueMeter()
    with open(logname, 'a') as f: #open and append
            f.write(str(network.module.model) + '\n')


    train_curve = []
    val_curve = []
    labels_generated_points = torch.Tensor(range(1, (opt.n_primitives+1)*(opt.num_points//opt.n_primitives)+1)).view(opt.num_points//opt.n_primitives,(opt.n_primitives+1)).transpose(0,1)
    labels_generated_points = (labels_generated_points)%(opt.n_primitives+1)
    labels_generated_points = labels_generated_points.contiguous().view(-1)


    for epoch in range(opt.nepoch):
        #TRAIN MODE
        train_loss.reset()
        network.module.model.train()
        torch.save(network.module.model.state_dict(), '%s/network%d.pth' % (dir_name, epoch))

        # learning rate schedule
        if epoch==20:
            optimizer = optim.Adam(network.module.model.parameters(), lr = lrate/10.0)
        if epoch==40:
            optimizer = optim.Adam(network.module.model.parameters(), lr = lrate/100.0)

        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            id, input, gt = data
            input = input.float().cuda()
            gt = gt.float().cuda()
            input = input.transpose(2,1).contiguous()

            output1, output2, emd_dcd1, emd_dcd2, expansion_penalty  = network(input, gt.contiguous(), 0.005, 50)
            
            loss_net = emd_dcd1.mean() + emd_dcd2.mean() + expansion_penalty.mean() * 0.1
            train_loss.update(emd_dcd2.mean().item())
            optimizer.step()

            if i % 10 == 0:
                idx = random.randint(0, input.size()[0] - 1)
                vis.scatter(X = gt.contiguous()[idx].data.cpu()[:, :3],
                        win = 'TRAIN_GT',
                        opts = dict(
                            title = id[idx],
                            markersize = 2,
                            ),
                        )
                vis.scatter(X = input.transpose(2,1).contiguous()[idx].data.cpu(),
                        win = 'TRAIN_INPUT',
                        opts = dict(
                            title = id[idx],
                            markersize = 2,
                            ),
                        )
                vis.scatter(X = output1[idx].data.cpu(),
                        Y = labels_generated_points[0:output1.size(1)],
                        win = 'TRAIN_COARSE',
                        opts = dict(
                            title= id[idx],
                            markersize=2,
                            ),
                        )
                vis.scatter(X = output2[idx].data.cpu(),
                        win = 'TRAIN_OUTPUT',
                        opts = dict(
                            title= id[idx],
                            markersize=2,
                            ),
                        )
            if opt.loss ==1:
                print(opt.env + ' train [%d: %d/%d]  emd1: %f emd2: %f expansion_penalty: %f' %(epoch, i, len_dataset/opt.batchSize, emd_dcd1.mean().item(), emd_dcd2.mean().item(), expansion_penalty.mean().item()))
            elif opt.loss ==2:
                print(opt.env + ' train [%d: %d/%d]  dcd1: %f dcd2: %f expansion_penalty: %f' %(epoch, i, len_dataset/opt.batchSize, emd_dcd1.mean().item(), emd_dcd2.mean().item(), expansion_penalty.mean().item()))
            else:
                raise NotImplementedError("Not implemented yet")
        train_curve.append(train_loss.avg)

        # VALIDATION
        if (epoch + 1) % 5 == 0:
            val_loss.reset()
            network.module.model.eval()
            with torch.no_grad():
                for i, data in enumerate(dataloader_test, 0):
                    id, input, gt = data
                    input = input.float().cuda()
                    gt = gt.float().cuda()
                    input = input.transpose(2,1).contiguous()
                    output1, output2, emd1, emd2, expansion_penalty  = network(input, gt.contiguous(), 0.004, 3000)
                    val_loss.update(emd2.mean().item())
                    idx = random.randint(0, input.size()[0] - 1)
                    vis.scatter(X = gt.contiguous()[idx].data.cpu()[:, :3],
                            win = 'VAL_GT',
                            opts = dict(
                                title = id[idx],
                                markersize = 2,
                                ),
                            )
                    vis.scatter(X = input.transpose(2,1).contiguous()[idx].data.cpu(),
                            win = 'VAL_INPUT',
                            opts = dict(
                                title = id[idx],
                                markersize = 2,
                                ),
                            )
                    vis.scatter(X = output1[idx].data.cpu(),
                            Y = labels_generated_points[0:output1.size(1)],
                            win = 'VAL_COARSE',
                            opts = dict(
                                title= id[idx],
                                markersize=2,
                                ),
                            )
                    vis.scatter(X = output2[idx].data.cpu(),
                            win = 'VAL_OUTPUT',
                            opts = dict(
                                title= id[idx],
                                markersize=2,
                                ),
                            )
                    if opt.loss ==1:
                        print(opt.env + ' train [%d: %d/%d]  emd1: %f emd2: %f expansion_penalty: %f' %(epoch, i, len_dataset/opt.batchSize, emd_dcd1.mean().item(), emd_dcd2.mean().item(), expansion_penalty.mean().item()))
                    elif opt.loss ==2:
                        print(opt.env + ' train [%d: %d/%d]  dcd1: %f dcd2: %f expansion_penalty: %f' %(epoch, i, len_dataset/opt.batchSize, emd_dcd1.mean().item(), emd_dcd2.mean().item(), expansion_penalty.mean().item()))
                    else:
                        raise NotImplementedError("Not implemented yet")
        val_curve.append(val_loss.avg)

        vis.line(X=np.column_stack((np.arange(len(train_curve)),np.arange(len(val_curve)))),
                     Y=np.column_stack((np.array(train_curve),np.array(val_curve))),
                     win='loss',
                     opts=dict(title="emd", legend=["train_curve" + opt.env, "val_curve" + opt.env], markersize=2, ), )
        vis.line(X=np.column_stack((np.arange(len(train_curve)),np.arange(len(val_curve)))),
                     Y=np.log(np.column_stack((np.array(train_curve),np.array(val_curve)))),
                     win='log',
                     opts=dict(title="log_emd", legend=["train_curve"+ opt.env, "val_curve"+ opt.env], markersize=2, ), )
        
        log_table = {
          "train_loss" : train_loss.avg,
          "val_loss" : val_loss.avg,
          "epoch" : epoch,
          "lr" : lrate,
          "bestval" : best_val_loss,

        }
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(log_table) + '\n')

        print('saving net...')
        torch.save(network.module.model.state_dict(), '%s/network.pth' % (dir_name))
