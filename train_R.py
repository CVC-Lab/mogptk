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
import numpy as np
import pdb
# from models.mosm.create_point_level_dataset import prepare_point_ds_mogptk
batch_size = 512


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



def _to_kernel_format(dataset, X, Y=None, is_multioutput=True):
        """
        Return the data vectors in the format used by the kernels. If Y is not passed, than only X data is returned.

        Returns:
            numpy.ndarray: X data of shape (data_points,input_dims). If the kernel is multi output, an additional input dimension is prepended with the channel indices.
            numpy.ndarray: Y data of shape (data_points,1).
            numpy.ndarray: Original but normalized X data. Only if no Y is passed.
        """
        # import pdb
        # pdb.set_trace()
        x = np.concatenate(X, axis=0)
        if is_multioutput:
            chan = [j * np.ones(len(X[j])) for j in range(len(X))]
            chan = np.concatenate(chan).reshape(-1, 1)
            x = np.concatenate([chan, x], axis=1)
        if Y is None:
            return x

        Y = list(Y) # shallow copy
        for j, channel_y in enumerate(Y):
            Y[j] = dataset[j].Y_transformer.forward(Y[j], X[j])
        y = np.concatenate(Y, axis=0).reshape(-1, 1)
        return x, y

def get_torch_dataloader(dataset, batch_size, shuffle=True):
    x, y = dataset.get_train_data()
    # # Assuming you want to pad the time series to a common length
    # common_length = max(len(ts) for ts in x)
    # # Pad time series
    # px = [np.pad(ts, pad_width=((0, common_length - len(ts)), (0, 0))) for ts in x]
    # common_length = max(len(ts) for ts in y)
    # # Pad time series
    # py = [np.pad(ts, pad_width=((0, common_length - len(ts)))) for ts in y]
    # px = np.concatenate(px, -1)
    # py = np.array(py).transpose(1, 0)

    px, py = _to_kernel_format(dataset, x, y)
    train_ds = TensorDataset(torch.from_numpy(px), torch.from_numpy(py))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    return train_loader

data_path = "./datasets/data/CAVE"
dataset = CAVEDataset(data_path, None, mode="train")
test_dataset = CAVEDataset(data_path, None, mode="test")
train_ds = prepare_point_ds_mogptk(dataset=dataset, num_points=None)
test_ds = prepare_point_ds_mogptk(dataset=test_dataset)
train_loader = get_torch_dataloader(train_ds, batch_size=batch_size, shuffle=True)
# train_x, train_y = prepare_point_ds(dataset=dataset, num_points=None)
# train_loader = TensorDataset(train_x, train_y)
# train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
# pdb.set_trace()
mogptk.gpr.use_gpu(0)
method = 'Adam'
lr = 0.02
iters = 500
# mogptk.gpr.use_cpu(0)
# mogptk.gpr.use_single_precision()
# pdb.set_trace()
# inducing_points = torch.randint(0, len(train_ds), 32)
num_inducing = 4**3
mosm = mogptk.MOSM(dataset=train_ds, train_loader=train_loader, inference=mogptk.model.Hensman(inducing_points=num_inducing),Q=1)
mosm.init_parameters(method='LS')
# pdb.set_trace()
mosm.train(method=method, lr=lr, iters=iters, verbose=True, jit=False)
# mosm.print_parameters()
# mosm.plot_prediction(transformed=True)
# error = mogptk.error(mosm, per_channel=True)[0]
# pdb.set_trace()
mosm.save("./artifacts/mosm_train_0")