"""Self-supervised Poisson denoising via exact binomial thinning (PRL).

Does not reuse deepinv's R2RLoss/PoissonNoise recorruption machinery: that
recorrupts with continuous additive noise, which breaks the discrete Poisson
structure of photon counts. Thinning a Poisson(mu) count via
Binomial(count, p) is itself exactly Poisson(p * mu) -- exact, not an
approximation.

Convention (matches deepinv.physics.noise.PoissonNoise, which is what
actually generates y for FMDD training):

    y = gamma * Poisson(x / gamma)

so the realized photon count is recovered as `count = y / gamma`, and a
LARGER gamma means MORE noise (fewer photons). To simulate additional
degradation on top of an already gamma_tn-noisy measurement y, its
recovered count is thinned down to an effective gamma_t > gamma_tn, with
retention probability q = gamma_tn / gamma_t (equivalently gamma_t = gamma_tn / q).
"""

import torch
import torch.nn as nn


class PoissonThinningLoss(nn.Module):
    """Poisson Reconstruction Loss (PRL) trained on binomially-thinned inputs.

    gamma_tn: Poisson gain of the input data y (== cfg.gamma).
    q:        thinning retention probability, in (0, 1). gamma_t = gamma_tn / q > gamma_tn.
    """

    def __init__(self, gamma_tn: float, q: float, eps: float = 1e-6):
        super().__init__()
        assert 0.0 < q < 1.0, f"q must be in (0, 1), got {q}"
        self.gamma_tn = gamma_tn
        self.q = q
        self.gamma_t = gamma_tn / q
        self.eps = eps

    def adapt_model(self, model: nn.Module) -> nn.Module:
        return _ThinningWrapper(model, self.q, self.gamma_tn, self.gamma_t)

    def forward(self, x_est, y, physics=None, model=None):
        # L_PRL(x, x_hat) = x*log(x/x_hat) + x_hat - x, with x*log(x/x_hat) := 0 where x == 0
        x_hat = x_est.clamp_min(self.eps)
        term = torch.where(
            y > 0,
            y * torch.log(y.clamp_min(self.eps) / x_hat),
            torch.zeros_like(y),
        )
        return (term + x_hat - y).mean()


class _ThinningWrapper(nn.Module):
    def __init__(self, model: nn.Module, q: float, gamma_tn: float, gamma_t: float):
        super().__init__()
        self.model = model  # kept as "model": train.py does model.model.parameters() / .set_context()
        self.q = q
        self.gamma_tn = gamma_tn
        self.gamma_t = gamma_t

    def _thin(self, y: torch.Tensor) -> torch.Tensor:
        count_tn = (y / self.gamma_tn).round().clamp_min(0)
        count_t = torch.distributions.Binomial(total_count=count_tn, probs=self.q).sample()
        return count_t * self.gamma_t

    def forward(self, y, physics=None, update_parameters=False):
        z = self._thin(y)
        x_hat = self.model(z)
        if update_parameters:
            return x_hat  # used by criterion(x_est, y, physics, model) during train / val-loss
        # The PRL minimizer is E[y|z] = q*z + (1-q)*E[x|z]; solve for E[x|z]. Same as
        # (gamma_t*x_hat - gamma_tn*z)/(gamma_t - gamma_tn): x_hat and z live in the rescaled
        # [0,1] domain, so gamma must weight both terms or the output is off by 1/gamma.
        return (x_hat - self.q * z) / (1.0 - self.q)
