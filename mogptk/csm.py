import numpy as np

from .config import logger
from .dataset import DataSet
from .model import Model, Exact
from .kernels import CrossSpectralKernel, MixtureKernel

class CSM(Model):
    """
    Cross Spectral Mixture kernel [1] with Q components and Rq latent functions. The parameters will be randomly instantiated, use init_parameters() to initialize the parameters to reasonable values for the current dataset.

    Args:
        dataset (mogptk.dataset.DataSet): DataSet object of data for all channels.
        Q (int, optional): Number of components.
        Rq (int, optional): Number of subcomponents.
        model: Gaussian Process model to use, such as mogptk.Exact.
        name (str, optional): Name of the model.

    Attributes:
        dataset: The associated mogptk.DataSet.
        model: The mogptk.kernels.Model.
        kernel: The mogptk.kernels.Kernel.

    Examples:

    >>> import numpy as np
    >>> import mogptk
    >>> 
    >>> t = np.linspace(0, 10, 100)
    >>> y1 = np.sin(0.5 * t)
    >>> y2 = 2.0 * np.sin(0.2 * t)
    >>> 
    >>> dataset = mogptk.DataSet(t, [y1, y2])
    >>> model = mogptk.CSM(dataset, Q=2)
    >>> model.init_parameters()
    >>> model.train()
    >>> model.predict()
    >>> dataset.plot()

    [1] K.R. Ulrich et al, "GP Kernels for Cross-Spectrum Analysis", Advances in Neural Information Processing Systems 28, 2015
    """
    def __init__(self, dataset, Q=1, Rq=1, model=Exact(), name="CSM"):
        if not isinstance(dataset, DataSet):
            dataset = DataSet(dataset)

        spectral = CrossSpectralKernel(
            output_dims=dataset.get_output_dims(),
            input_dims=dataset.get_input_dims()[0],
            Rq=Rq,
        )
        kernel = MixtureKernel(spectral, Q)

        super(CSM, self).__init__(dataset, kernel, model, name)
        self.Q = Q
        self.Rq = Rq
        if issubclass(type(model), Exact):
            self.model.noise.assign(0.0, lower=0.0, trainable=False)  # handled by MultiOutputKernel
    
    def init_parameters(self, method='BNSE', sm_init='BNSE', sm_method='Adam', sm_iters=100, sm_params={}, sm_plot=False):
        """
        Initialize kernel parameters.

        The initialization can be done in two ways, the first by estimating the PSD via 
        BNSE (Tobar 2018) and then selecting the greater Q peaks in the estimated spectrum,
        the peaks position, magnitude and width initialize the mean, magnitude and variance
        of the kernel respectively.
        The second way is by fitting independent Gaussian process for each channel, each one
        with SM kernel, using the fitted parameters for initial values of the multioutput kernel.

        In all cases the noise is initialized with 1/30 of the variance 
        of each channel.

        Args:
            method (str, optional): Method of estimation, such as BNSE, LS, or SM.
            sm_init (str, optional): Parameter initialization strategy for SM initialization.
            sm_method (str, optional): Optimization method for SM initialization.
            sm_iters (str, optional): Number of iterations for SM initialization.
            sm_params (object, optional): Additional parameters for PyTorch optimizer.
            sm_plot (bool): Show the PSD of the kernel after fitting SM.
        """

        output_dims = self.dataset.get_output_dims()

        if not method.lower() in ['bnse', 'ls', 'sm']:
            raise ValueError("valid methods of estimation are BNSE, LS, and SM")

        if method.lower() == 'bnse':
            amplitudes, means, variances = self.dataset.get_bnse_estimation(self.Q)
        elif method.lower() == 'ls':
            amplitudes, means, variances = self.dataset.get_lombscargle_estimation(self.Q)
        else:
            amplitudes, means, variances = self.dataset.get_sm_estimation(self.Q, method=sm_init, optimizer=sm_method, iters=sm_iters, params=sm_params, plot=sm_plot)
        if len(amplitudes) == 0:
            logger.warning('{} could not find peaks for MOSM'.format(method))
            return

        # flatten output_dims and mixtures
        channels = [channel for channel, amplitude in enumerate(amplitudes) for q in range(amplitude.shape[0])]
        amplitudes = [amplitude[q,:] for amplitude in amplitudes for q in range(amplitude.shape[0])]
        means = [mean[q,:] for mean in means for q in range(mean.shape[0])]
        variances = [variance[q,:] for variance in variances for q in range(variance.shape[0])]
        idx = np.argsort([amplitude.mean() for amplitude in amplitudes])[::-1]

        constant = np.random.random((output_dims, self.Q, self.Rq))
        for q in range(len(idx)):
            i = idx[q]
            channel = channels[i]
            constant[channel,q % self.Q,:] = amplitudes[i].mean()
            if q < self.Q:
                self.model.kernel[q].mean.assign(means[i])
                self.model.kernel[q].variance.assign(variances[i] * 5.0)

        # normalize proportional to channel variance
        for i, channel in enumerate(self.dataset):
            _, y = channel.get_train_data(transformed=True)
            if 0.0 < constant[i,:,:].sum():
                constant[i,:,:] = constant[i,:,:] / constant[i,:,:].sum() * y.var() * 2

        for q in range(self.Q):
            self.model.kernel[q].amplitude.assign(constant[:,q,:])

        noise = np.empty((output_dims,))
        for i, channel in enumerate(self.dataset):
            _, y = channel.get_train_data(transformed=True)
            noise[i] = y.var() / 30.0
        for q in range(self.Q):
            self.model.kernel[q].noise.assign(noise)
