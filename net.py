import torch
from compressai.models import ScaleHyperprior
from compressai.entropy_models import EntropyBottleneck, GaussianConditional, EntropyModel
from sga import Quantizator_SGA
import numpy as np

class EntropyBottleneckNoQuant(EntropyBottleneck):
    def __init__(self, channels):
        super().__init__(channels)
        self.sga = Quantizator_SGA()

    def forward(self, x_quant):
        perm = np.arange(len(x_quant.shape))
        perm[0], perm[1] = perm[1], perm[0]
        # Compute inverse permutation
        inv_perm = np.arange(len(x_quant.shape))[np.argsort(perm)]
        x_quant = x_quant.permute(*perm).contiguous()
        shape = x_quant.size()
        x_quant = x_quant.reshape(x_quant.size(0), 1, -1)
        likelihood = self._likelihood(x_quant)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        # Convert back to input tensor shape
        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()
        return likelihood

class GaussianConditionalNoQuant(GaussianConditional):
    def __init__(self, scale_table):
        super().__init__(scale_table=scale_table)

    def forward(self, x_quant, scales, means):
        likelihood = self._likelihood(x_quant, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return likelihood

class ScaleHyperpriorSGA(ScaleHyperprior):
    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)
        self.entropy_bottleneck = EntropyBottleneckNoQuant(N)
        self.gaussian_conditional  = GaussianConditionalNoQuant(None)
        self.sga = Quantizator_SGA()

    def quantize(self, inputs, mode, means=None, it=None, tot_it=None):
        if means is not None:
            inputs = inputs - means
        if mode == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            outputs = inputs + noise
        elif mode == "round":
            outputs = torch.round(inputs)
        elif mode == "sga":
            outputs = self.sga(inputs, it, "training", tot_it)
        else:
            assert(0)
        if means is not None:
            outputs = outputs + means
        return outputs

    def forward(self, x, mode, y_in=None, z_in=None, it=None, tot_it=None):
        if mode == "init":
            y = self.g_a(x)
            z = self.h_a(torch.abs(y))
        else:
            y = y_in
            z = z_in
        if mode == "init" or mode == "round":
            y_hat = self.quantize(y, "round")
            z_hat = self.quantize(z, "round")
        elif mode == "noise":
            y_hat = self.quantize(y, "noise")
            z_hat = self.quantize(z, "noise")
        elif mode =="sga":
            y_hat = self.quantize(y, "sga", None, it, tot_it)
            z_hat = self.quantize(z, "sga", None, it, tot_it)
        else:
            assert(0)
        z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_likelihoods = self.gaussian_conditional(y_hat, scales_hat, None)
        x_hat = self.g_s(y_hat)
        return {
            "y": y.detach().clone(),
            "z": z.detach().clone(), 
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
