
from einops import rearrange, reduce, repeat, pack, unpack
from datasets.cave_dataset import CAVEDataset
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import torch
import tqdm
import pdb
from . import mogptk
from .mogptk.data import Data
from models.metrics import (
    compare_mpsnr,
    compare_mssim,
    find_rmse,
    # compare_sam,
    compare_ergas
)
import pdb
from torchmetrics import SpectralAngleMapper
from torchmetrics import ErrorRelativeGlobalDimensionlessSynthesis as ERGAS
sam = SpectralAngleMapper()
ergas = ERGAS(ratio=1/8)


def prepare_point_ds_mogptk(dataset, num_points=None):
    Xt = []
    Yt = []

    for i, items in enumerate(dataset):
        y, z, x_gt, seg_map,  _  = items
        seg_map = rearrange(torch.from_numpy(np.array(seg_map)), 
                            'h w c -> c w h')#.flip(0)
        
        xt, _ = pack([z, seg_map], '* w h')
        xt = rearrange(xt, 'c h w -> (h w) c')
        yt = rearrange(x_gt, 'c h w -> (h w) c')
        Xt.append(xt)
        Yt.append(yt)
        if i + 1 > num_points:
            break


    Xt, ps = pack(Xt, "* c") 
    Yt, ps = pack(Yt, "* c")
    ds = mogptk.DataSet()
    
    # shuffle here
    shuf_idx = torch.randperm(num_points, device='cpu')
    Xt = Xt[shuf_idx]
    Yt = Yt[shuf_idx]

    for c in range(Yt.shape[1]):
        d = Data(Xt[:num_points, :], Yt[:num_points, c])
        # d = Data(Xt, Yt[:, c])
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
    px, py = _to_kernel_format(dataset, x, y)
    train_ds = TensorDataset(torch.from_numpy(px), torch.from_numpy(py))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    return train_loader


def predict(model, test_loader):
    with torch.no_grad():

        # The output of the model is a multitask MVN, where both the data points
        # and the tasks are jointly distributed
        # To compute the marginal predictive NLL of each data point,
        # we will call `to_data_independent_dist`,
        # which removes the data cross-covariance terms from the distribution.
        means, vars = [], []
        lowers, uppers = [], []

        for tx, ty in tqdm.tqdm(test_loader):
            #tx = tx.cuda() # [4096, 3]
            # pdb.set_trace()
            x_pred, mean, lower, upper = model.predict(tx)
            # mean, var = preds.mean.mean(0), preds.variance.mean(0)
            mean = np.stack(mean, -1)
            lower = np.stack(lower, -1)
            upper = np.stack(upper, -1)
            
            means.append(mean)
            lowers.append(lower)
            uppers.append(upper)

    means, _ = pack(means, '* d')
    lowers, _ = pack(lowers, '* d')
    uppers, _ = pack(uppers, '* d')
    means = torch.from_numpy(means).to(torch.float32)
    lowers = torch.from_numpy(lowers).to(torch.float32)
    uppers = torch.from_numpy(uppers).to(torch.float32)
    return means, lowers, uppers


mogptk.gpr.use_gpu(0)
num_inducing = 4**3
save_path = "./artifacts/mosm_train_material_Q1"
data_path = "./datasets/data/CAVE"
batch_size = 64 * 64
# dataset = CAVEDataset(data_path, None, mode="train")
test_dataset = CAVEDataset(data_path, None, mode="test")
# train_ds = prepare_point_ds_mogptk(dataset=dataset, num_points=None)
# train_loader = get_torch_dataloader(train_ds, batch_size=batch_size, shuffle=True)

# mosm = mogptk.MOSM(dataset=train_ds, 
#                    train_loader=train_loader, 
#                    inference=mogptk.model.Hensman(inducing_points=num_inducing),
#                    Q=1)

mosm = mogptk.model.LoadModel(save_path)
# pdb.set_trace()


# load image from dataset convert to pointwise then extract mean and convert back to image
total_psnr, total_ssim, total_rmse, total_sam, total_ergas =0,0,0,0,0
for items in test_dataset:
    y, z, x_gt, seg_map,  max_vals  = items
    seg_map = rearrange(torch.from_numpy(np.array(seg_map)), 
                            'h w c -> c w h')#.flip(0)
    z, _ = pack([z, seg_map], '* w h')
    test_x = rearrange(z, 'c h w -> (h w) c')
    test_y = rearrange(x_gt, 'c h w -> (h w) c')
    
    test_ds = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    # Make predictions
    mean, lower, upper = predict(mosm, test_loader)
    pred_x = torch.reshape(mean, x_gt.shape)
    pdb.set_trace()

    
    # print(pred_x.shape)
    # pdb.set_trace()
    x_gt = rearrange(x_gt, 'c h w -> h w c').detach().cpu().numpy()
    pred_x = rearrange(pred_x, 'c h w -> h w c').detach().cpu().numpy()
    total_ssim += compare_mssim(x_gt, pred_x)
    rmse,  mse, rmse_per_band = find_rmse(x_gt, pred_x)
    total_rmse += rmse
    total_psnr += compare_mpsnr(x_gt, pred_x, mse)
    total_sam += torch.nan_to_num(sam(torch.from_numpy(x_gt).permute(2, 0, 1)[None, ...], 
                            torch.from_numpy(pred_x).permute(2, 0, 1)[None, ...]), nan=0, posinf=1.0) * (180/torch.pi)
    total_ergas += compare_ergas(x_gt, pred_x, 8, rmse_per_band)
    

opt = f"""## Metric scores:
psnr:{total_psnr/len(test_dataset)},
ssim:{total_ssim/len(test_dataset)},
rmse:{total_rmse/len(test_dataset)},
sam:{total_sam/len(test_dataset)},
ergas:{total_ergas/len(test_dataset)},
"""
print(opt)


