import mogptk
import torch
import numpy as np
import pandas as pd
import pdb
torch.manual_seed(1)

from torch.utils.data import TensorDataset, DataLoader
print(mogptk.__file__)
mogptk.gpr.use_gpu(0)
column_names = ['EUR/USD', 'CAD/USD', 'JPY/USD', 'GBP/USD', 'CHF/USD',
                'AUD/USD', 'HKD/USD','NZD/USD', 'KRW/USD','MXN/USD']

dataset = mogptk.DataSet()
for names in column_names:
    dataset.append(mogptk.LoadCSV('examples/data/currency_exchange/final_dataset.csv',
                                    x_col='Date', y_col=names))

dataset.filter('2017-01-03', '2018-01-01')

# Preprocess by randomly removing points and detrending
for i, channel in enumerate(dataset):
    channel.transform(mogptk.TransformDetrend)
    channel.transform(mogptk.TransformNormalize())
    channel.remove_randomly(pct=0.3)
    
    if i not in [0, 2, 5]:
        channel.remove_range('2017-11-17', None)
    
# simulate sensor failure
dataset[1].remove_range('2017-03-31', '2017-05-01')
dataset[2].remove_range('2017-12-28', None)
dataset[3].remove_range('2017-07-20', '2017-09-08')
dataset[4].remove_range(None, '2017-01-31')
dataset[5].remove_range('2017-12-28', None)
dataset[7].remove_range(None, '2017-01-31')

n_trials = 3
Q = 3
init_method = 'LS'
method = 'Adam'
lr = 0.1
iters = 1000


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

def get_torch_dataloader(dataset, batch_size):
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
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    return train_loader

mosm_models = []
mosm_mae = np.zeros((n_trials,10))
mosm_rmse = np.zeros((n_trials,10))
mosm_mape = np.zeros((n_trials,10))

# experiment trials
for n in range(n_trials):
    mosm_dataset = dataset.copy()
    for i, channel in enumerate(mosm_dataset):
        channel.remove_randomly(pct=0.3)
    
    train_loader = get_torch_dataloader(mosm_dataset, 1067)
    print('\nTrial', n+1, 'of', n_trials)
    num_inducing = (1 + 1)**3
    mosm = mogptk.MOSM(dataset=mosm_dataset, train_loader=train_loader, 
                       inference=mogptk.model.Hensman(inducing_points=num_inducing),
                       Q=Q)
    mosm.init_parameters(method=init_method)
    # pdb.set_trace()
    mosm.train(method=method, lr=lr, iters=iters, verbose=True, jit=False)
    # mosm.train(method=method, lr=lr, iters=iters, verbose=True)
    mosm_models.append(mosm)
    print('=' * 50)
    
    error = mogptk.error(mosm, per_channel=True)[0]
    mosm_mae[n,:] = np.array([item['MAE'] for item in error])
    mosm_rmse[n,:] = np.array([item['RMSE'] for item in error])
    mosm_mape[n,:] = np.array([item['MAPE'] for item in error])