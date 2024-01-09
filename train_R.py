import sys
import torch
from torch.optim import Adam
import torch.nn.functional as F
from einops import rearrange, reduce, repeat, pack, unpack
from .mogptk.data import Data
from . import mogptk
torch.manual_seed(1)
from datasets.cave_dataset import CAVEDataset
from torch.utils.data import TensorDataset, DataLoader
import pdb
# from models.mosm.create_point_level_dataset import prepare_point_ds_mogptk
batch_size = 256


def prepare_point_ds_mogptk(dataset, num_points=None):
    Xt = []
    Yt = []
    if not num_points:
        num_points = len(dataset)

    for i, items in enumerate(dataset):
        y, z, x_gt, _  = items
        xt = rearrange(z, 'c h w -> (h w) c')
        yt = rearrange(x_gt, 'c h w -> (h w) c')
        Xt.append(xt)
        Yt.append(yt)
        if i + 1 > num_points:
            break


    Xt, ps = pack(Xt, "* c") 
    Yt, ps = pack(Yt, "* c")
    ds = mogptk.DataSet()
    
    for c in range(Yt.shape[1]):
        d = Data(Xt[:num_points, :], Yt[:num_points, c])
        ds.append(d)
    return ds

def prepare_point_ds(dataset, num_points=None):
    Xt = []
    Yt = []
    for items in dataset:
        y, z, x_gt, _  = items
        xt = rearrange(z, 'c h w -> (h w) c')
        yt = rearrange(x_gt, 'c h w -> (h w) c')
        Xt.append(xt)
        Yt.append(yt)

    Xt, ps_x = pack(Xt, "* c") 
    Yt, ps_y = pack(Yt, "* c")
    # pdb.set_trace()
    return Xt, Yt

data_path = "./datasets/data/CAVE"
dataset = CAVEDataset(data_path, None, mode="train")
test_dataset = CAVEDataset(data_path, None, mode="test")
train_ds = prepare_point_ds_mogptk(dataset=dataset, num_points=None)
test_ds = prepare_point_ds_mogptk(dataset=test_dataset)

train_x, train_y = prepare_point_ds(dataset=dataset, num_points=None)
train_loader = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
# pdb.set_trace()
method = 'Adam'
lr = 0.02
iters = 5
# mogptk.gpr.use_cpu(0)
# mogptk.gpr.use_single_precision()
# pdb.set_trace()
# inducing_points = torch.randint(0, len(train_ds), 32)
num_inducing = 4**3
mosm = mogptk.MOSM(dataset=train_ds, train_loader=train_loader, inference=mogptk.model.Hensman(inducing_points=num_inducing),Q=1)
mosm.init_parameters(method='LS')
# pdb.set_trace()
mosm.train(method=method, lr=lr, iters=iters, verbose=True, jit=False)
mosm.print_parameters()
mosm.plot_prediction(transformed=True)