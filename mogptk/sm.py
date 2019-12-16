import numpy as np
from .model import model
from .kernels import SpectralMixture, sm_init, Noise
from scipy.stats import norm
import matplotlib.pyplot as plt

def _estimate_from_sm(dataset, Q, estimate='BNSE', method='BFGS', maxiter=2000, plot=False, fix_means=False):
    """
    Estimate kernel param with single ouput GP-SM

    Args:
        dataset (mogptk.DataSet): Data class instance.
        Q (int): Number of components.
        estimate (str): Method to estimate, 'BNSE', 'LS' or 'Random'
        method (str): Optimization method, either 'Adam' or any
            scipy optimizer.
        maxiter (int): Maximum number of iteration.
        plot (bool): If true plot the kernel PSD of each channel.
        fix_means(bool): Fix spectral means to zero in trainning.

    Returns: 
        params[q][name][output dim][input dim]
    """
    input_dims = dataset.get_input_dims()[0]
    output_dims = dataset.get_output_dims()

    params = []
    for q in range(Q):
        params.append({
            'weight': np.empty((input_dims, output_dims)),
            'mean': np.empty((input_dims, output_dims)),
            'scale': np.empty((input_dims, output_dims)),
        })

    for channel in range(output_dims):
        for i in range(input_dims):  # TODO one SM per channel
            sm = SM(data[channel], Q)
            sm.estimate_params(estimate)

            if fix_means:
                sm.set_param(0, 'mixture_means', np.zeros(Q, input_dims))
                sm.set_param(0, 'mixture_scales', sm.get_param(0, 'mixture_scale') * 100.0)
                sm.fix_param('mixture_means')

            sm.train(method=method, maxiter=maxiter, tol=1e-50)

            if plot:
                nyquist = dataset[channel].get_nyquist_estimation()
                means = sm._get_param_across('mixture_means')
                weights = sm._get_param_across('mixture_weights')
                scales = sm._get_param_across('mixture_scales')
                plot_spectrum(means, scales, weights=weights, nyquist=nyquist, title=data[channel].name)

            for q in range(Q):
                params[q]['weight'][i,channel] = sm.get_param(0, 'mixture_weights')[q]
                params[q]['mean'][i,channel] = sm.get_param(0, 'mixture_means')[q,:] * np.pi * 2
                params[q]['scale'][i,channel] = sm.get_param(0, 'mixture_scales')[:,q]

    return params

class SM(model):
    """
    Single output GP with Spectral mixture kernel.
        
    Args:
        data (mogptk.DataSet): DataSet object of data for all channels. Only one channel allowed for SM.
        Q (int): Number of components to use.
        name (string): Name of the model.
        likelihood (gpflow.likelihoods): Likelihood to use from GPFlow, if None a default exact inference Gaussian likelihood is used.
        variational (bool): If True, use variational inference to approximate function values as Gaussian. If False it will use Monte Carlo Markov Chain (default).
        sparse (bool): If True, will use sparse GP regression. Defaults to False.
        like_params (dict): Parameters to GPflow likelihood.
    """
    def __init__(self, dataset, Q=1, name="SM", likelihood=None, variational=False, sparse=False, like_params={}):
        model.__init__(self, name, dataset)
        self.Q = Q

        if self.dataset.get_output_dims() != 1:
            raise Exception("single output Spectral Mixture kernel can only take one output dimension in the data")

        with self.graph.as_default():
            with self.session.as_default():
                kernel = SpectralMixture(
                    self.dataset.get_input_dims()[0],
                    self.Q,
                )
                self._build(kernel, likelihood, variational, sparse, like_params)

    def estimate_params(self, method='BNSE'):
        """
        Estimate parameters of kernel from data using different methods.

        Kernel parameters can be initialized using 3 heuristics using the train data:

        'random': (Taken from phd thesis from Andrew wilson 2014) is taking the inverse
            of lengthscales drawn from truncated Gaussian N(0, max_dist^2), the
            means drawn from Unif(0, 0.5 / minimum distance between two points),
            and the mixture weights by taking the stdv of the y values divided by the
            number of mixtures.
        'LS'_ is using Lomb Scargle periodogram for estimating the PSD,
            and using the first Q peaks as the means and mixture weights.
        'BNSE': third is using the BNSE (Tobar 2018) to estimate the PSD 
            and use the first Q peaks as the means and mixture weights.

        In all cases the noise is initialized with 1/30 of the variance of each channel.

        *** Only for single input dimension for each channel.
        """

        if method not in ['random', 'LS', 'BNSE']:
            raise Exception("possible methods are 'random', 'LS' and 'BNSE' (see documentation).")

        if method == 'random':
            x, y = self.dataset[0].get_data()
            weights, means, scales = sm_init(x, y, self.Q)
            for q in range(self.Q):
                self.set_param(0, 'mixture_weights', weights)
                self.set_param(0, 'mixture_means', np.array(means))
                self.set_param(0, 'mixture_scales', np.array(scales.T))

        elif method == 'LS':
            amplitudes, means, variances = self.dataset[0].get_ls_estimation(self.Q)
            if len(amplitudes) == 0:
                logging.warning('LS could not find peaks for SM')
                return

            for q in range(self.Q):
                self.set_param(q, 'mixture_weights', amplitudes.mean(axis=0) / amplitudes.mean())
                self.set_param(q, 'mixture_means', means.T)
                self.set_param(q, 'mixture_scales', variances * 2)

        elif method == 'BNSE':
            amplitudes, means, variances = self.dataset[0].get_bnse_estimation(self.Q)
            if np.sum(amplitudes) == 0.0:
                logging.warning('BNSE could not find peaks for SM')
                return

            for q in range(self.Q):
                self.set_param(0, 'mixture_weights', amplitudes.mean(axis=0) / amplitudes.mean())
                self.set_param(0, 'mixture_means', means.T)
                self.set_param(0, 'mixture_scales', variances * 2)

    def plot_psd(self, figsize=(10, 4), title='', log_scale=False):
        """
        Plot power spectral density of single output GP-SM.
        """
        means = self.get_param(0, 'mixture_means')
        weights = self.get_param(0, 'mixture_weights')
        scales = self.get_param(0, 'mixture_scales').T
        
        # calculate bounds
        x_low = norm.ppf(0.001, loc=means, scale=scales).min()
        x_high = norm.ppf(0.99, loc=means, scale=scales).max()
        
        x = np.linspace(x_low, x_high + 1, 1000)
        psd = np.zeros_like(x)

        fig, axes = plt.subplots(1, 1, figsize=figsize)
        for q in range(self.Q):
            single_psd = weights[q] * norm.pdf(x, loc=means[q], scale=scales[q])
            axes.plot(x, single_psd, '--', lw=1.2, c='xkcd:strawberry', zorder=2)
            axes.axvline(means[q], ymin=0.001, ymax=0.1, lw=2, color='grey')
            psd = psd + single_psd
            
        # symmetrize PSD
        if psd[x<0].size != 0:
            psd = psd + np.r_[psd[x<0][::-1], np.zeros((x>=0).sum())]
            
        axes.plot(x, psd, lw=2.5, c='r', alpha=0.7, zorder=1)
        axes.set_xlim(0, x[-1] + 0.1)
        if log_scale:
            axes.set_yscale('log')
        axes.set_xlabel(r'$\omega$')
        axes.set_ylabel('PSD')
        axes.set_title(title)
