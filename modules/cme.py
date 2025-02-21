from dataclasses import dataclass
import json
from importlib.resources import files
from typing import Any

import numpy as np
import torch
from torch import nn, Tensor

from modules.exprel import exprel


@dataclass
class CMEParams:
    n: int
    optim: str
    a: list[float]
    b: list[float]
    c: float
    omega: float
    phi: list[float]
    lognorm: float
    mu1: float
    mu2: float
    cv2: float


def load_cme_params() -> list[CMEParams]:
    data = files("modules").joinpath("iltcme.json").read_text()
    # data = pkgutil.get_data(__name__, 'iltcme.json')
    cme_params_dicts: list[dict[str, Any]] = json.loads(data)

    cme_params = [CMEParams(**p) for p in cme_params_dicts]
    return cme_params


class CME(nn.Module):
    _eta: Tensor
    s: Tensor
    tau_stars: Tensor

    cme_params = load_cme_params()  # load once, shared with all instances

    def __init__(
        self,
        tau_min: float,
        tau_max: float,
        n_taus: int,
        max_fn_evals: int,
        g: int,
        batch_first: bool = False,
    ) -> None:
        # super().__init__(tau_min, tau_max, n_taus, g, batch_first)
        super().__init__()
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.n_taus = n_taus
        self.max_fn_evals = max_fn_evals
        self.g = g
        self.batch_first = batch_first

        tau_stars = torch.tensor(np.geomspace(tau_min, tau_max, n_taus))
        self.register_buffer("tau_stars", tau_stars, persistent=False)

        # adapted from: https://github.com/ghorvath78/iltcme/blob/master/ipython_ilt.ipynb
        # find the most steep CME satisfying maxFnEvals
        best_p = CME.cme_params[0]
        for p in CME.cme_params:
            if p.cv2 < best_p.cv2 and p.n + 1 <= max_fn_evals:
                best_p = p
        a, b, c = best_p.a, best_p.b, best_p.c
        n, omega, mu1 = best_p.n, best_p.omega, best_p.mu1

        eta_real = torch.tensor((c, *a), dtype=torch.float64)
        eta_imag = torch.tensor((0, *b), dtype=torch.float64)
        eta = torch.complex(eta_real, eta_imag) * mu1
        self.register_buffer("_eta", eta, persistent=False)

        beta_real = torch.ones(n + 1).double()
        beta_imag = torch.arange(n + 1).double() * omega
        beta = torch.complex(beta_real, beta_imag) * mu1
        # self.register_buffer("_beta", beta, persistent=False)

        s = torch.outer(1 / self.tau_stars, beta)
        self.register_buffer("s", s, persistent=False)

        self.fn_evals = n

    # @property
    # def s(self) -> Tensor:
    #     return self._s

    def forward(
        self,
        fs: Tensor,  # (batch, seq, feat) if self.batch_first else (seq, batch, feat)
        F: Tensor | None = None,  # (batch, feat, s, s2)
        alphas: Tensor | None = None,  #  <shape same as fs.shape>
    ) -> tuple[
        Tensor,  # til_F or Fs: (batch, seq, feat, taustar)
        Tensor,  # F: (batch, feat, s, s2)
    ]:
        alphas = alphas if alphas is not None else torch.ones_like(fs)

        if fs.shape != alphas.shape:
            raise ValueError(
                f"fs and alphas must have the same shape, but "
                f"have shapes {fs.shape} and {alphas.shape}."
            )

        if self.batch_first:
            # (batch, seq, feat) -> (seq, batch, feat)
            fs = fs.transpose(0, 1)
            alphas = alphas.transpose(0, 1)

        if F is None:
            # Generate initial F
            _, n_batch, n_feat = fs.shape
            F = fs.new_zeros((n_batch, n_feat, *self.s.shape))

        # === Forward Laplace Transform ===
        s_mul_a = self.s * alphas[..., None, None]  # outer product
        hh = torch.exp(-s_mul_a)  # hidden -> hidden
        ih = exprel(-s_mul_a)  # input -> hidden
        b = fs[..., None, None] * ih

        Fs = torch.empty_like(hh)
        for i in range(len(Fs)):
            # equivalent to: F = F * [e**(-s*a)] + [f * (e**(-s*a)-1)/(-s*a)]
            F = F * hh[i] + b[i]
            Fs[i] = F

        # Fs = self.Fs_transform_hook(Fs)

        # === Inverse Laplace transform ===
        til_fs = torch.inner(self._eta, Fs).real / self.tau_stars

        # if g=1, multiply by tau_stars and divide by number of s per til_f
        til_fs = til_fs * (self.tau_stars / self.fn_evals) ** self.g

        if self.batch_first:
            # (seq, batch, feat, taustar) -> (batch, seq, feat, taustar)
            til_fs = til_fs.transpose(0, 1)

        return til_fs, F
    
    def get_translations(
            self, 
            F: Tensor,  # (batch, feat, s, s2)
        ) -> Tensor:  # (batch, delta, feat, taustar)
        # generates spectrum of memory projections for specified timestep

        # R = torch.outer(-self.tau_stars, self.s).exp()  # decay amount, translates til_f
        alphas = self.tau_stars
        s_mul_a = self.s * alphas[..., None, None]  # outer product
        R = torch.exp(-s_mul_a)

        delta_F = torch.einsum("dsv, bfsv -> bdfsv", R, F)  # v = s2
        # delta_til_f = delta_F @ self.post

        delta_til_f = torch.inner(self._eta, delta_F).real / self.tau_stars

        # if g=1, multiply by tau_stars and divide by number of s per til_f
        delta_til_f = delta_til_f * (self.tau_stars / self.fn_evals) ** self.g

        return delta_til_f  # [:, self.k:-self.k, :, :]
