#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core algorithmic module for Bayesian-MaxEnt PSD inversion (v3.2.1).

v3.2.1 improvements:
  ① Adds optional smoothness prior on PSD in u-space:
    lambda * ||Delta^2 u||^2, where u = log(x / m).
  ② Keeps legacy behavior when smooth_lambda = 0.
  ③ Includes smoothness penalty in MAP score and Laplace scalar term.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import traceback as _tb
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import minimize

Array = np.ndarray


# ---------------------------------------------------------------------
# Utility: grids
# ---------------------------------------------------------------------
def make_log_radius_bins(r_min: float, r_max: float, nbins: int) -> Tuple[Array, Array]:
    if r_min <= 0 or r_max <= 0:
        raise ValueError("r_min and r_max must be > 0")
    if r_max <= r_min:
        raise ValueError("r_max must be > r_min")
    edges = np.logspace(np.log10(r_min), np.log10(r_max), nbins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return edges, centers


# ---------------------------------------------------------------------
# Sphere form factor
# ---------------------------------------------------------------------
def _phi_sphere(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-3
    xs = x[~small]
    if xs.size:
        out[~small] = 3.0 * (np.sin(xs) - xs * np.cos(xs)) / (xs ** 3)
    if np.any(small):
        x0 = x[small]
        out[small] = 1.0 - (x0 ** 2) / 10.0 + (x0 ** 4) / 280.0 - (x0 ** 6) / 15120.0
    return out


def kernel_sphere_volume_fraction(q: Array, r: Array) -> Array:
    q = np.asarray(q, dtype=float).reshape(-1)
    r = np.asarray(r, dtype=float).reshape(-1)
    V = 4.0 * np.pi * (r ** 3) / 3.0
    qr = np.outer(q, r)
    Phi = _phi_sphere(qr)
    K = (Phi ** 2) * V
    return K


# ---------------------------------------------------------------------
# Structure factor models
# ---------------------------------------------------------------------
def structure_factor_unity(q: Array, params: Dict[str, float] | None = None) -> Array:
    return np.ones_like(q, dtype=float)


def structure_factor_oz(q: Array, params: Dict[str, float]) -> Array:
    xi = float(params["xi"])
    q = np.asarray(q, dtype=float)
    return 1.0 / (1.0 + (q * xi) ** 2)


def structure_factor_hardsphere_py(q: Array, params: Dict[str, float]) -> Array:
    R = float(params["R_hs"])
    eta = float(params["eta"])
    if not (0.0 < eta < 0.49):
        raise ValueError("eta must be between 0 and 0.49 for PY hard-sphere model.")
    q = np.asarray(q, dtype=float)
    x = q * 2.0 * R

    alpha = (1.0 + 2.0 * eta) ** 2 / (1.0 - eta) ** 4
    beta = -6.0 * eta * (1.0 + eta / 2.0) ** 2 / (1.0 - eta) ** 4
    gamma = 0.5 * eta * alpha

    term1 = np.empty_like(x)
    term2 = np.empty_like(x)
    term3 = np.empty_like(x)

    small = np.abs(x) < 0.1
    xs = x[~small]
    if xs.size:
        sinx = np.sin(xs); cosx = np.cos(xs)
        term1[~small] = (sinx - xs * cosx) / xs ** 2
        term2[~small] = (2.0 * xs * sinx + (2.0 - xs ** 2) * cosx - 2.0) / xs ** 3
        term3num = (-xs ** 4) * cosx + 4.0 * (
            (3.0 * xs ** 2 - 6.0) * cosx + (xs ** 3 - 6.0 * xs) * sinx + 6.0)
        term3[~small] = term3num / xs ** 5
    if np.any(small):
        x0 = x[small]
        term1[small] = x0 / 3.0 - x0 ** 3 / 30.0 + x0 ** 5 / 840.0
        term2[small] = x0 / 4.0 - x0 ** 3 / 36.0 + x0 ** 5 / 960.0
        term3[small] = x0 / 6.0 - x0 ** 3 / 48.0 + x0 ** 5 / 1200.0

    G = alpha * term1 + beta * term2 + gamma * term3
    S = np.empty_like(x)
    zero = np.abs(x) < 1e-12
    S[zero] = 1.0 / alpha
    S[~zero] = 1.0 / (1.0 + 24.0 * eta * G[~zero] / x[~zero])
    return S


def structure_factor_effective_2017(q: Array, params: Dict[str, float]) -> Array:
    q = np.asarray(q, dtype=float)
    log_a = float(np.clip(params["log_a"], -30.0, 30.0))
    a = float(np.exp(log_a))
    b = float(np.clip(params["b"], -10.0, 10.0))
    S_hs = structure_factor_hardsphere_py(q, {"R_hs": params["R_hs"], "eta": params["eta"]})
    q_safe = np.maximum(q, np.min(q[q > 0]) * 0.5 if np.any(q > 0) else 1.0)
    return a * (q_safe ** b) + S_hs


STRUCTURE_FACTOR_MODELS: Dict[str, Callable[[Array, Dict[str, float] | None], Array]] = {
    "unity": structure_factor_unity,
    "Ornstein-Zernike": structure_factor_oz,
    "hard_sphere_py": structure_factor_hardsphere_py,
    "effective_2017": structure_factor_effective_2017,
}

SF_REQUIRED_PARAMS: Dict[str, List[str]] = {
    "unity": [],
    "Ornstein-Zernike": ["xi"],
    "hard_sphere_py": ["R_hs", "eta"],
    "effective_2017": ["log_a", "b", "R_hs", "eta"],
}


# ---------------------------------------------------------------------
# Likelihoods / misfit
# ---------------------------------------------------------------------
def cost_gaussian(I_pred: Array, y: Array, sigma: Array) -> float:
    r = (I_pred - y) / sigma
    return 0.5 * float(np.sum(r * r))


def dcost_dI_gaussian(I_pred: Array, y: Array, sigma: Array) -> Array:
    return (I_pred - y) / (sigma ** 2)


def cost_poisson_scaled(I_pred, y, scale, eps=1e-10, mu_max=1e300):
    I_pred = np.asarray(I_pred, float)
    y = np.asarray(y, float)
    mu_c = np.clip(I_pred * scale, eps, mu_max)
    y_c = y * scale
    return float(np.sum(mu_c - y_c * np.log(mu_c)))


def dcost_dI_poisson_scaled(I_pred, y, scale, eps=1e-10, mu_max=1e300):
    I_pred = np.asarray(I_pred, float)
    y = np.asarray(y, float)
    mu_c = np.clip(I_pred * scale, eps, mu_max)
    y_c = y * scale
    return scale * (1.0 - y_c / mu_c)


def poisson_deviance_scaled(I_pred, y, scale, eps=1e-10, mu_max=1e300):
    I_pred = np.asarray(I_pred, float)
    y = np.asarray(y, float)
    mu_c = np.clip(I_pred * scale, eps, mu_max)
    y_c = y * scale
    mpos = y_c > 0
    term = np.zeros_like(y_c)
    term[mpos] = y_c[mpos] * (np.log(y_c[mpos]) - np.log(mu_c[mpos]))
    return float(2.0 * np.sum(mu_c - y_c + term))


def estimate_poisson_scale_from_sigma(y: Array, sigma: Array) -> float:
    y = np.asarray(y, float)
    sigma = np.asarray(sigma, float)
    y_safe = np.maximum(y, 0.0)
    mask = (y_safe > 0) & np.isfinite(y_safe) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(mask):
        return 1.0
    s = y_safe[mask] / (sigma[mask] ** 2)
    return float(np.median(s))


# ---------------------------------------------------------------------
# Numerically stable log-det
# ---------------------------------------------------------------------
def _stable_logdet_spd(H: Array, jitter0: float = 1e-12, max_tries: int = 10) -> Optional[float]:
    """Cholesky-based log-determinant with progressive jittering."""
    H = 0.5 * (H + H.T)
    n = max(H.shape[0], 1)
    scale = max(abs(float(np.trace(H))) / n, 1.0)
    jitter = jitter0 * scale
    I_eye = np.eye(n)
    for _ in range(max_tries):
        try:
            L = np.linalg.cholesky(H + jitter * I_eye)
            return 2.0 * float(np.sum(np.log(np.diag(L))))
        except np.linalg.LinAlgError:
            jitter *= 10.0
            if jitter > 1e-2 * scale:
                break
    # Fallback: use eigenvalues
    try:
        eigvals = np.linalg.eigvalsh(H)
        pos = eigvals[eigvals > 1e-30]
        if pos.size > 0:
            return float(np.sum(np.log(pos)))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------
# MaxEnt MAP solver (inner loop)
# ---------------------------------------------------------------------
@dataclass
class MaxEntResult:
    f_map: Array
    I_map: Array
    alpha: float
    S: float
    C: float
    score: float
    success: bool
    message: str
    nfev: int
    smooth_cost: float = 0.0
    D: Optional[float] = None
    I_map_trunc: Optional[Array] = None


@dataclass
class MaxEntConfig:
    likelihood: Literal["gaussian", "poisson"] = "gaussian"
    poisson_scale: Optional[float | Array] = None
    poisson_scale_clip: Tuple[float, float] = (1e-12, 1e12)
    C_target: Optional[float] = None
    D_target: Optional[float] = None
    alpha_mode: Literal["chi2", "fixed", "deviance"] = "chi2"
    alpha_fixed: float = 1.0
    alpha_bounds: Tuple[float, float] = (1e-6, 1e6)
    alpha_tol: float = 1e-3
    max_bisect_iter: int = 40
    opt_maxiter: int = 2000
    opt_ftol: float = 1e-10
    opt_gtol: float = 1e-8
    eps_I: float = 1e-10
    u_bounds: Tuple[float, float] = (-60.0, 60.0)
    sigma_rel_floor: float = 0.0
    sigma_abs_floor: float = 0.0
    # Optional smoothness prior on u=log(x/m): lambda * ||Delta^2 u||^2
    smooth_lambda: float = 0.1
    n_r: int = 90


class MaxEntMAP:
    def __init__(self,
                 A: Array,
                 y: Array,
                 sigma: Optional[Array],
                 m: Array,
                 cfg: MaxEntConfig):
        self.A = np.asarray(A, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self.sigma = None if sigma is None else np.asarray(sigma, dtype=float).reshape(-1)
        self.m = np.asarray(m, dtype=float).reshape(-1)
        self.cfg = cfg

        if self.A.shape[0] != self.y.size:
            raise ValueError("A rows must match length of y")
        if self.A.shape[1] != self.m.size:
            raise ValueError("A cols must match length of m")

        if cfg.likelihood == "gaussian":
            if self.sigma is None:
                raise ValueError("Gaussian likelihood requires sigma.")
            if np.any(self.sigma <= 0):
                raise ValueError("sigma must be >0.")
            rel = float(cfg.sigma_rel_floor)
            abs0 = float(cfg.sigma_abs_floor)
            if (rel > 0.0) or (abs0 > 0.0):
                y_abs = np.abs(self.y)
                self.sigma = np.sqrt(self.sigma ** 2 + (rel * y_abs) ** 2 + abs0 ** 2)

        if np.any(self.m <= 0):
            raise ValueError("Default model m must be strictly positive.")

        self.smooth_lambda = max(0.0, float(getattr(cfg, "smooth_lambda", 0.1)))
        self._smooth_L = None
        self._smooth_L_diag = None
        if self.smooth_lambda > 0.0 and int(cfg.n_r) >= 3:
            nr = int(cfg.n_r)
            d2 = np.zeros((nr - 2, nr), dtype=float)
            ridx = np.arange(nr - 2)
            d2[ridx, ridx] = 1.0
            d2[ridx, ridx + 1] = -2.0
            d2[ridx, ridx + 2] = 1.0
            self._smooth_L = d2.T @ d2
            self._smooth_L_diag = np.diag(self._smooth_L).copy()

        self.poisson_scale = None
        self.poisson_mask = None

        if cfg.likelihood == "poisson":
            if self.sigma is None and cfg.poisson_scale is None:
                raise ValueError("Poisson needs sigma to infer scale, or set cfg.poisson_scale manually.")

            y_raw = self.y
            if cfg.poisson_scale is not None:
                s0 = np.asarray(cfg.poisson_scale, float)
                if np.ndim(s0) == 0:
                    self.poisson_scale = float(np.clip(float(s0), *cfg.poisson_scale_clip))
                    self.poisson_mask = np.isfinite(y_raw) & (y_raw >= 0.0)
                else:
                    s_vec = np.asarray(s0, float).reshape(-1)
                    if s_vec.size != self.y.size:
                        raise ValueError("poisson_scale vector length must match y length.")
                    lo, hi = cfg.poisson_scale_clip
                    s_vec = np.clip(s_vec, lo, hi)
                    self.poisson_mask = (np.isfinite(y_raw) & (y_raw >= 0.0)
                                         & np.isfinite(s_vec) & (s_vec > 0))
                    self.poisson_scale = s_vec
            else:
                sig = np.asarray(self.sigma, float)
                lo, hi = cfg.poisson_scale_clip
                mask = np.isfinite(y_raw) & (y_raw >= 0.0) & np.isfinite(sig) & (sig > 0)
                s_pos = y_raw[mask] / (sig[mask] ** 2)
                s_med = float(np.median(s_pos)) if s_pos.size else 1.0
                s_vec = np.full_like(y_raw, s_med)
                s_vec[mask] = s_pos
                s_vec = np.clip(s_vec, lo, hi)
                self.poisson_mask = (np.isfinite(y_raw) & (y_raw >= 0.0)
                                     & np.isfinite(s_vec) & (s_vec > 0))
                self.poisson_scale = s_vec

        self.A_full = self.A
        self.y_full = self.y

        # Static data truncation.
        if cfg.likelihood == "poisson" and self.poisson_mask is not None:
            self.poisson_scale_full = np.copy(self.poisson_scale) if np.ndim(self.poisson_scale) > 0 else self.poisson_scale
            self.y = self.y[self.poisson_mask]
            self.A = self.A[self.poisson_mask, :]
            if np.ndim(self.poisson_scale) > 0:
                self.poisson_scale = self.poisson_scale[self.poisson_mask]
            self.poisson_mask = None
        elif cfg.likelihood == "gaussian":
            self.sigma_full = self.sigma.copy()
            mask = np.isfinite(self.y) & np.isfinite(self.sigma) & (self.sigma > 0)
            self.y = self.y[mask]
            self.A = self.A[mask, :]
            self.sigma = self.sigma[mask]

    # ---------------------------------------------------------------------
    def _smooth_penalty_u(self, u: Array) -> Tuple[float, Array]:
        nr = int(self.cfg.n_r)
        if (self.smooth_lambda <= 0.0) or (self._smooth_L is None) or (nr < 3):
            return 0.0, np.zeros(nr, dtype=float)
        u_psd = np.asarray(u[:nr], dtype=float)
        Lu = self._smooth_L @ u_psd
        cost = 0.5 * self.smooth_lambda * float(u_psd @ Lu)
        grad = self.smooth_lambda * Lu
        return cost, grad

    # ---------------------------------------------------------------------
    def _cost_and_grad_u(self, u: Array, alpha: float) -> Tuple[float, Array, Array, float, float]:
        u = np.asarray(u, float)
        x = self.m * np.exp(u)
        I_pred = self.A @ x
        nr = self.cfg.n_r

        if self.cfg.likelihood == "gaussian":
            C = cost_gaussian(I_pred, self.y, self.sigma)
            dC_dI = dcost_dI_gaussian(I_pred, self.y, self.sigma)
        elif self.cfg.likelihood == "poisson":
            D_here = poisson_deviance_scaled(I_pred, self.y, scale=self.poisson_scale, eps=self.cfg.eps_I)
            C = 0.5 * D_here
            dC_dI = dcost_dI_poisson_scaled(I_pred, self.y, scale=self.poisson_scale, eps=self.cfg.eps_I)
        else:
            raise ValueError(f"Unknown likelihood: {self.cfg.likelihood}")

        dC_dx = self.A.T @ dC_dI
        # Entropy only over PSD bins (not background/powerlaw)
        S = float(np.sum(x[:nr] - self.m[:nr] - x[:nr] * u[:nr]))
        smooth_cost, smooth_grad_u = self._smooth_penalty_u(u)
        J = float(C - alpha * S + smooth_cost)

        grad_u = np.zeros_like(u)
        grad_u[:nr] = (dC_dx[:nr] + alpha * u[:nr]) * x[:nr] + smooth_grad_u
        # Extra params: likelihood-only gradient by design.
        # Note: the weak diagonal term used in laplace_score() is a curvature
        # regularizer for logdet stability, not part of MAP optimization.
        if len(u) > nr:
            grad_u[nr:] = dC_dx[nr:] * x[nr:]

        return J, grad_u, x, S, C

    # ---------------------------------------------------------------------
    def solve_for_alpha(self, alpha: float, u0: Optional[Array] = None) -> MaxEntResult:
        if u0 is None:
            u0 = np.zeros_like(self.m)

        def fun(u):
            J, grad_u, *_ = self._cost_and_grad_u(u, alpha)
            return J, grad_u

        bounds = [self.cfg.u_bounds] * len(u0)
        res = minimize(fun, x0=u0, method="L-BFGS-B", jac=True, bounds=bounds,
                       options={"maxiter": self.cfg.opt_maxiter,
                                "ftol": self.cfg.opt_ftol,
                                "gtol": self.cfg.opt_gtol})

        u_hat = np.asarray(res.x, dtype=float)
        J, grad_u, x_hat, S, C = self._cost_and_grad_u(u_hat, alpha)
        smooth_cost, _ = self._smooth_penalty_u(u_hat)

        score = float(alpha * S - C - smooth_cost)

        I_hat_full = self.A_full @ x_hat
        I_hat_trunc = self.A @ x_hat

        D = None
        if self.cfg.likelihood == "poisson":
            D = poisson_deviance_scaled(I_hat_trunc, self.y, scale=self.poisson_scale, eps=self.cfg.eps_I)

        return MaxEntResult(
            f_map=x_hat, I_map=I_hat_full, alpha=float(alpha),
            S=float(S), C=float(C), score=float(score),
            success=bool(res.success), message=str(res.message),
            nfev=int(res.nfev) if hasattr(res, "nfev") else -1,
            smooth_cost=float(smooth_cost), D=D,
            I_map_trunc=I_hat_trunc
        )

    # ---------------------------------------------------------------------
    def laplace_score(self, x: Array, I_pred: Array, alpha: float,
                      C: float, S: float) -> float:
        """Laplace approximation to log marginal likelihood.

        v3.2.1 keeps weak regularization for extra parameters
        (background, powerlaw_amp) to stabilize the Hessian log-determinant.
        This term is intentionally curvature-only (used here for Laplace
        stability) and is not included in _cost_and_grad_u MAP optimization.
        """
        x = np.asarray(x, float)
        if np.any(x <= 0):
            return -np.inf

        if self.cfg.likelihood == "gaussian":
            w = 1.0 / (self.sigma ** 2)
            AW = self.A * w[:, None]
            H_C = self.A.T @ AW
        else:
            I_pred_trunc = np.asarray(I_pred, float).reshape(-1)
            mu = np.maximum(I_pred_trunc, self.cfg.eps_I)
            # Use expected Fisher information for scaled Poisson:
            # E[y] = mu  =>  w = scale / mu
            if np.ndim(self.poisson_scale) > 0:
                w = self.poisson_scale / mu
            else:
                w = float(self.poisson_scale) / mu
            AW = self.A * w[:, None]
            H_C = self.A.T @ AW

        H = H_C.copy()
        nr = self.cfg.n_r

        # PSD bins: entropy prior contributes alpha / x_j to diagonal
        idx = np.arange(nr)
        H[idx, idx] += alpha / np.maximum(x[:nr], 1e-300)

        smooth_cost = 0.0
        if (self.smooth_lambda > 0.0) and (self._smooth_L is not None):
            u_psd = np.log(np.maximum(x[:nr], 1e-300) / self.m[:nr])
            Lu = self._smooth_L @ u_psd
            smooth_cost = 0.5 * self.smooth_lambda * float(u_psd @ Lu)
            # Approximate x-space diagonal curvature induced by u-space prior.
            H[idx, idx] += (
                self.smooth_lambda * self._smooth_L_diag
            ) / np.maximum(x[:nr], 1e-300) ** 2

        # Add weak diagonal curvature for non-PSD extras (background, powerlaw)
        # only in Laplace Hessian to stabilize logdet; MAP solve is unchanged.
        # Prior variance scale is approximately (10 * x_j)^2.
        for j in range(nr, len(x)):
            x_j = max(abs(float(x[j])), 1e-30)
            H[j, j] += 1.0 / (100.0 * x_j ** 2)

        logdet = _stable_logdet_spd(H)
        if logdet is None:
            return -np.inf
        return alpha * S - C - smooth_cost - 0.5 * logdet

    # ---------------------------------------------------------------------
    def solve(self, u0: Optional[Array] = None) -> MaxEntResult:
        """Solve for optimal alpha via bisection, then return MAP result.

        This method profiles out alpha and should be used by outer search.
        """
        if self.cfg.alpha_mode == "fixed":
            return self.solve_for_alpha(self.cfg.alpha_fixed, u0=u0)

        if self.cfg.alpha_mode == "deviance":
            if self.cfg.likelihood != "poisson":
                raise ValueError("alpha_mode='deviance' is intended for Poisson likelihood.")
            if self.cfg.D_target is None:
                raise ValueError("D_target must be set for alpha_mode='deviance'.")

            alpha_lo, alpha_hi = self.cfg.alpha_bounds
            u_curr = np.zeros_like(self.m) if u0 is None else u0.copy()

            res_lo = self.solve_for_alpha(alpha_lo, u0=u_curr)
            u_lo = np.log(np.maximum(res_lo.f_map, 1e-300) / self.m)

            res_hi = self.solve_for_alpha(alpha_hi, u0=u_curr)
            u_hi = np.log(np.maximum(res_hi.f_map, 1e-300) / self.m)

            D_lo, D_hi = res_lo.D, res_hi.D
            Dt = self.cfg.D_target
            if D_lo is None or D_hi is None:
                raise RuntimeError("Poisson deviance was not computed.")
            if D_hi <= Dt:
                return res_hi
            if D_lo > Dt:
                return res_lo

            best = res_lo
            log_lo, log_hi = math.log(alpha_lo), math.log(alpha_hi)
            for _ in range(self.cfg.max_bisect_iter):
                log_mid = 0.5 * (log_lo + log_hi)
                alpha_mid = math.exp(log_mid)
                # Warm-start from best known solution
                u_warm = np.log(np.maximum(best.f_map, 1e-300) / self.m)
                res_mid = self.solve_for_alpha(alpha_mid, u0=u_warm)
                D_mid = res_mid.D
                if D_mid is None:
                    raise RuntimeError("Poisson deviance was not computed.")
                if abs(D_mid - Dt) < abs(best.D - Dt):
                    best = res_mid
                if D_mid <= Dt:
                    log_lo = log_mid
                    res_lo = res_mid
                else:
                    log_hi = log_mid
                    res_hi = res_mid
                if (log_hi - log_lo) < self.cfg.alpha_tol:
                    break
            return best

        # chi2 mode
        if self.cfg.alpha_mode != "chi2":
            raise ValueError(f"Unknown alpha_mode: {self.cfg.alpha_mode}")
        if self.cfg.C_target is None:
            raise ValueError("C_target must be set for alpha_mode='chi2'.")

        alpha_lo, alpha_hi = self.cfg.alpha_bounds
        if not (alpha_lo > 0 and alpha_hi > alpha_lo):
            raise ValueError("Invalid alpha_bounds.")

        u_curr = np.zeros_like(self.m) if u0 is None else u0.copy()
        res_lo = self.solve_for_alpha(alpha_lo, u0=u_curr)
        u_lo = np.log(np.maximum(res_lo.f_map, 1e-300) / self.m)
        res_hi = self.solve_for_alpha(alpha_hi, u0=u_curr)
        u_hi = np.log(np.maximum(res_hi.f_map, 1e-300) / self.m)
        Ct = self.cfg.C_target
        if res_hi.C <= Ct:
            return res_hi
        if res_lo.C > Ct:
            return res_lo

        log_lo, log_hi = math.log(alpha_lo), math.log(alpha_hi)
        best = res_lo
        for _ in range(self.cfg.max_bisect_iter):
            log_mid = 0.5 * (log_lo + log_hi)
            alpha_mid = math.exp(log_mid)
            u_mid0 = u_lo if abs(res_lo.C - Ct) < abs(res_hi.C - Ct) else u_hi
            res_mid = self.solve_for_alpha(alpha_mid, u0=u_mid0)
            if abs(res_mid.C - Ct) < abs(best.C - Ct):
                best = res_mid
            if res_mid.C <= Ct:
                log_lo = log_mid
                res_lo = res_mid
                u_lo = np.log(np.maximum(res_lo.f_map, 1e-300) / self.m)
            else:
                log_hi = log_mid
                res_hi = res_mid
                u_hi = np.log(np.maximum(res_hi.f_map, 1e-300) / self.m)
            if (log_hi - log_lo) < self.cfg.alpha_tol:
                break
        return best


# ---------------------------------------------------------------------
# Outer hyper-parameter search
# ---------------------------------------------------------------------
@dataclass
class OuterSearchConfig:
    mode: Literal["none", "grid_search"] = "grid_search"
    bounds: Dict[str, Tuple[float, float]] = dataclasses.field(default_factory=dict)
    step_scale: Dict[str, float] = dataclasses.field(default_factory=dict)
    seed: int = 0
    grid_points: int = 20
    local_opt_top_k: int = 3
    local_opt_xtol: float = 1e-4
    local_opt_ftol: float = 1e-4
    # Two-stage grid refinement
    two_stage: bool = True
    fine_grid_factor: float = 0.25       # shrink bounds to ±25% around best
    fine_grid_points: int = 15           # points per dim in fine stage
    fine_grid_warm_start: bool = False   # keep v3.1-like behavior by default


def _in_bounds(theta: Dict[str, float], bounds: Dict[str, Tuple[float, float]]) -> bool:
    for k, (lo, hi) in bounds.items():
        if k not in theta:
            continue
        if not (lo <= theta[k] <= hi):
            return False
    return True


# ---------------------------------------------------------------------
# PSD Inversion config & result
# ---------------------------------------------------------------------
@dataclass
class PSDInversionConfig:
    r_min: float
    r_max: float
    n_r: int
    sf_model: str = "unity"
    sf_params0: Dict[str, float] = dataclasses.field(default_factory=dict)
    include_background: bool = True
    include_powerlaw: bool = False
    powerlaw_exponent: float = 4.0
    maxent: MaxEntConfig = dataclasses.field(default_factory=MaxEntConfig)
    outer: OuterSearchConfig = dataclasses.field(default_factory=OuterSearchConfig)
    default_total: float = 1.0
    default_background: Optional[float] = None
    default_powerlaw_amp: Optional[float] = None

    def __post_init__(self):
        self.maxent.n_r = self.n_r


@dataclass
class PSDInversionResult:
    edges: Array
    r_centers: Array
    psd_bins: Array
    extras: Dict[str, float]
    sf_params: Dict[str, float]
    alpha: float
    S: float
    C: float
    score: float
    I_fit: Array
    D: Optional[float] = None


# ---------------------------------------------------------------------
# Design matrix builder
# ---------------------------------------------------------------------
def _build_design_matrix(q: Array, K: Array, sf_model: str,
                         sf_params: Dict[str, float],
                         include_background: bool,
                         include_powerlaw: bool,
                         powerlaw_exponent: float) -> Tuple[Array, List[str]]:
    q = np.asarray(q, float)
    sf_params = sf_params if sf_params is not None else {}
    missing = [k for k in SF_REQUIRED_PARAMS.get(sf_model, []) if k not in sf_params]
    if missing:
        raise ValueError(f"SF model '{sf_model}' missing params: {missing}")
    sf_fun = STRUCTURE_FACTOR_MODELS[sf_model]
    S_q = sf_fun(q, sf_params)
    cols = [S_q[:, None] * K]
    names = [f"psd_{j}" for j in range(K.shape[1])]
    if include_background:
        cols.append(np.ones((q.size, 1)))
        names.append("background")
    if include_powerlaw:
        q_safe = np.maximum(q, np.min(q[q > 0]) * 0.5 if np.any(q > 0) else 1.0)
        cols.append((q_safe ** (-powerlaw_exponent)).reshape(-1, 1))
        names.append("powerlaw_amp")
    A = np.concatenate(cols, axis=1)
    return A, names


# ---------------------------------------------------------------------
# Multiprocessing worker (v3.2.1: alpha profiled out)
# ---------------------------------------------------------------------
_WORKER_CTX = {}


def _init_worker(q, K, sf_model, include_background, include_powerlaw,
                 powerlaw_exponent, y, sigma, m, cfg_maxent):
    global _WORKER_CTX
    _WORKER_CTX = dict(q=q, K=K, sf_model=sf_model,
                       include_background=include_background,
                       include_powerlaw=include_powerlaw,
                       powerlaw_exponent=powerlaw_exponent,
                       y=y, sigma=sigma, m=m, cfg_maxent=cfg_maxent)


def _global_worker(args):
    """Worker evaluation with alpha profiled by solver.solve()."""
    k_idx, theta_sf, u0 = args
    if theta_sf is None:
        return k_idx, None, None, None
    ctx = _WORKER_CTX

    try:
        A, names = _build_design_matrix(
            q=ctx["q"], K=ctx["K"], sf_model=ctx["sf_model"],
            sf_params=theta_sf,
            include_background=ctx["include_background"],
            include_powerlaw=ctx["include_powerlaw"],
            powerlaw_exponent=ctx["powerlaw_exponent"])
        solver = MaxEntMAP(A=A, y=ctx["y"], sigma=ctx["sigma"],
                           m=ctx["m"], cfg=ctx["cfg_maxent"])

        # Profile out alpha by bisection.
        res = solver.solve(u0=u0)

        # Compute Laplace evidence score
        I_for_score = res.I_map_trunc if res.I_map_trunc is not None else res.I_map
        sc = solver.laplace_score(res.f_map, I_for_score,
                                  res.alpha, res.C, res.S)
        sc += 0.5 * ctx["cfg_maxent"].n_r * math.log(max(res.alpha, 1e-300))
        res.score = sc

        return k_idx, res, names, theta_sf
    except Exception as e:
        # Print only first few worker errors per process to avoid log flooding.
        if not hasattr(_global_worker, "_err_count"):
            _global_worker._err_count = 0
        _global_worker._err_count += 1
        if _global_worker._err_count <= 3:
            print(f"[worker] Grid point {k_idx} failed: {e}", flush=True)
            _tb.print_exc()
        elif _global_worker._err_count == 4:
            print("[worker] Suppressing further error messages...", flush=True)
        return k_idx, None, None, None
# ---------------------------------------------------------------------
# Main inversion function
# ---------------------------------------------------------------------
def invert_psd(q: Array, y: Array, sigma: Optional[Array],
               cfg: PSDInversionConfig) -> PSDInversionResult:
    # Keep maxent dimensionality consistent with runtime n_r edits.
    cfg.maxent.n_r = cfg.n_r

    edges, r_centers = make_log_radius_bins(cfg.r_min, cfg.r_max, cfg.n_r)
    K = kernel_sphere_volume_fraction(q, r_centers)

    # Estimate default model m.
    s_i = np.sum(K, axis=1)
    y_arr = np.asarray(y, float)
    poisson_vector_scale = False
    if cfg.maxent.likelihood == "poisson":
        if cfg.maxent.poisson_scale is not None:
            s_scale = np.asarray(cfg.maxent.poisson_scale, float)
            w_i = s_scale / np.maximum(y_arr, 1e-12)
            poisson_vector_scale = (np.ndim(s_scale) > 0)
        else:
            if sigma is not None:
                w_i = 1.0 / np.maximum(np.asarray(sigma, float) ** 2, 1e-12)
            else:
                w_i = 1.0 / np.maximum(y_arr, 1e-12)
    else:
        if sigma is not None:
            w_i = 1.0 / np.maximum(np.asarray(sigma, float) ** 2, 1e-12)
        else:
            w_i = np.ones_like(y_arr)

    mask = (y_arr > 0) & np.isfinite(w_i) & (w_i > 0)
    if poisson_vector_scale:
        min_counts = 3.0
        good = mask & (y_arr > min_counts)
        if np.any(good):
            num = np.sum(w_i[good] * s_i[good] * y_arr[good])
            den = np.sum(w_i[good] * s_i[good] ** 2)
            m0 = num / den if den > 0 else cfg.default_total / cfg.n_r
        elif np.any(mask):
            num = np.sum(w_i[mask] * s_i[mask] * y_arr[mask])
            den = np.sum(w_i[mask] * s_i[mask] ** 2)
            m0 = num / den if den > 0 else cfg.default_total / cfg.n_r
        else:
            m0 = cfg.default_total / cfg.n_r
    elif np.any(mask):
        num = np.sum(w_i[mask] * s_i[mask] * y_arr[mask])
        den = np.sum(w_i[mask] * s_i[mask] ** 2)
        m0 = num / den if den > 0 else cfg.default_total / cfg.n_r
    else:
        m0 = cfg.default_total / cfg.n_r

    m_psd = np.full(cfg.n_r, m0, dtype=float)
    m_list = [m_psd]
    if cfg.include_background:
        bg0 = (max(float(np.median(y) * 0.01), 1e-12)
               if cfg.default_background is None else float(cfg.default_background))
        m_list.append(np.array([bg0], dtype=float))
    if cfg.include_powerlaw:
        if cfg.default_powerlaw_amp is None:
            q_safe = np.maximum(q, np.min(q[q > 0]) * 0.5 if np.any(q > 0) else 1.0)
            amp0 = max(float(np.median(y[:max(3, len(y) // 10)])
                             * (q_safe[0] ** cfg.powerlaw_exponent) * 0.1), 1e-12)
        else:
            amp0 = float(cfg.default_powerlaw_amp)
        m_list.append(np.array([amp0], dtype=float))
    m = np.concatenate(m_list)

    # Set targets when not explicitly provided.
    n_params = cfg.n_r + (1 if cfg.include_background else 0) + (1 if cfg.include_powerlaw else 0)
    if cfg.maxent.likelihood == "gaussian" and cfg.maxent.C_target is None:
        cfg.maxent.C_target = 0.5 * max(1.0, float(len(q)) - min(float(len(q)) * 0.5, float(n_params)))
    if cfg.maxent.likelihood == "poisson" and cfg.maxent.D_target is None:
        if cfg.maxent.alpha_mode == "chi2":
            cfg.maxent.alpha_mode = "deviance"
        n_eff = int(np.sum(np.isfinite(y_arr) & (y_arr >= 0)))
        nu_eff = min(float(n_eff) * 0.5, float(n_params))
        cfg.maxent.D_target = max(1.0, float(n_eff) - nu_eff)

    # Evaluation helper with alpha profiled by solver.solve().
    def evaluate(theta_sf: Dict[str, float],
                 u0: Optional[Array] = None) -> Tuple[MaxEntResult, List[str]]:
        """Build design matrix, solve with profiled-out alpha, compute score."""
        try:
            A, names = _build_design_matrix(
                q=q, K=K, sf_model=cfg.sf_model, sf_params=theta_sf,
                include_background=cfg.include_background,
                include_powerlaw=cfg.include_powerlaw,
                powerlaw_exponent=cfg.powerlaw_exponent)
            solver = MaxEntMAP(A=A, y=y, sigma=sigma, m=m, cfg=cfg.maxent)

            # Alpha is profiled out by solver.solve().
            res = solver.solve(u0=u0)

            # Compute Laplace evidence score
            I_for_score = res.I_map_trunc if res.I_map_trunc is not None else res.I_map
            sc = solver.laplace_score(res.f_map, I_for_score,
                                      res.alpha, res.C, res.S)
            sc += 0.5 * cfg.n_r * math.log(max(res.alpha, 1e-300))
            res.score = sc
            return res, names
        except Exception:
            y_arr = np.asarray(y, float)
            finite = np.isfinite(y_arr)
            if np.any(finite):
                y_min = float(np.min(y_arr[finite]))
                y_max = float(np.max(y_arr[finite]))
            else:
                y_min = float("nan")
                y_max = float("nan")
            print("[evaluate failed] sf_model=", cfg.sf_model, "theta=", theta_sf)
            print("  y min/max:", y_min, y_max)
            print("  y<0 count:", int(np.sum(y_arr < 0)))
            raise

    # Initial evaluation at user-supplied SF parameters.
    theta = dict(cfg.sf_params0)

    def _clip_theta(theta_dict, bounds):
        out = dict(theta_dict)
        for k, (lo, hi) in bounds.items():
            if k in out and not (lo <= out[k] <= hi):
                out[k] = float(np.clip(out[k], lo, hi))
                print(f"[outer] init {k} out of bounds, clipped to {out[k]}")
        return out

    if cfg.outer.mode != "none":
        theta = _clip_theta(theta, cfg.outer.bounds)
    res0, names0 = evaluate(theta, u0=None)

    best_res = res0
    best_theta = dict(theta)
    best_names = names0

    # ---------------------------------------------------------------------
    # Outer search
    # ---------------------------------------------------------------------
    if cfg.outer.mode == "none":
        # Keep alpha profiled by bisection even when outer search is disabled.
        pass

    elif cfg.outer.mode == "grid_search":
        import itertools

        # Search only structure-factor parameters.
        var_names = [k for k in cfg.outer.bounds.keys() if k != "log_alpha"]
        if not var_names:
            best_theta, best_res, best_names = theta, res0, names0
        else:
            # Build a 1-D grid for one parameter.
            def _make_grid_1d(name, lo, hi, n):
                lo, hi = float(lo), float(hi)
                if name in ("R_hs", "xi"):
                    lo = max(lo, 1e-9)
                    hi = max(hi, lo * (1 + 1e-9))
                    return np.logspace(np.log10(lo), np.log10(hi), n)
                elif name == "eta":
                    # Denser near middle of allowed range
                    return np.linspace(lo, hi, n)
                else:
                    return np.linspace(lo, hi, n)

            # Run grid evaluation (parallel when available).
            def _run_grid(bounds_dict, n_per_dim, stage_label="",
                          u0_seed: Optional[Array] = None):
                grids = []
                for k in var_names:
                    lo, hi = bounds_dict[k]
                    seq = _make_grid_1d(k, lo, hi, n_per_dim)
                    grids.append(seq)

                # Stream tasks via generator to avoid large intermediate lists.
                total = n_per_dim ** len(grids)
                print(f"[outer{stage_label}] Grid evaluating {total} points "
                      f"(alpha profiled out via bisection)...", flush=True)

                WIN_CAP = 61 if sys.platform.startswith("win") else (os.cpu_count() or 96)
                max_workers = min(max(1, total), WIN_CAP)
                main_mod = sys.modules.get("__main__", None)
                main_file = None if main_mod is None else getattr(main_mod, "__file__", None)
                interactive_main = (
                    main_file is None
                    or str(main_file).startswith("<")
                    or (not os.path.exists(str(main_file)))
                )

                def _task_gen():
                    for k_idx, vals in enumerate(itertools.product(*grids)):
                        th_prop = dict(theta)
                        for j, k in enumerate(var_names):
                            th_prop[k] = float(vals[j])
                        yield (k_idx, th_prop, u0_seed)

                def _run_serial(msg: str):
                    print(f"[outer{stage_label}] {msg}", flush=True)
                    _init_worker(q, K, cfg.sf_model, cfg.include_background,
                                 cfg.include_powerlaw, cfg.powerlaw_exponent,
                                 y, sigma, m, cfg.maxent)
                    results = []
                    print_freq = max(1, total // 20)
                    for i, task in enumerate(_task_gen()):
                        r = _global_worker(task)
                        results.append(r)
                        if (i + 1) % print_freq == 0 or (i + 1) == total:
                            print(f"  [grid{stage_label}] {i+1}/{total} "
                                  f"({100.0*(i+1)/total:.1f}%)", flush=True)
                    return results

                if interactive_main or total <= 1:
                    reason = "Serial evaluation in interactive/small-task mode."
                    results = _run_serial(reason)
                else:
                    print(f"[outer{stage_label}] Spawning {max_workers} workers...", flush=True)
                    try:
                        with ProcessPoolExecutor(
                            max_workers=max_workers,
                            initializer=_init_worker,
                            initargs=(q, K, cfg.sf_model, cfg.include_background,
                                      cfg.include_powerlaw, cfg.powerlaw_exponent,
                                      y, sigma, m, cfg.maxent)
                        ) as executor:
                            chunksize = max(1, total // (max_workers * 4))
                            results = []
                            print_freq = max(1, total // 20)
                            for i, r in enumerate(executor.map(_global_worker, _task_gen(),
                                                               chunksize=chunksize)):
                                results.append(r)
                                if (i + 1) % print_freq == 0 or (i + 1) == total:
                                    print(f"  [grid{stage_label}] {i+1}/{total} "
                                          f"({100.0*(i+1)/total:.1f}%)", flush=True)
                    except (BrokenProcessPool, RuntimeError, OSError) as e:
                        msg = f"Parallel failed ({e}); falling back to serial."
                        results = _run_serial(msg)

                # r = (k_idx, res, names, theta_sf) — theta_sf echoed by worker
                valid = [r for r in results
                         if r[1] is not None and np.isfinite(r[1].score)]
                valid.sort(key=lambda x: x[1].score, reverse=True)
                return valid

            # Stage 1: coarse grid.
            coarse_n = cfg.outer.grid_points
            if coarse_n is None or coarse_n <= 0:
                coarse_n = 20

            coarse_bounds = {k: cfg.outer.bounds[k] for k in var_names}
            valid_coarse = _run_grid(coarse_bounds, coarse_n,
                                     stage_label=" coarse")

            if not valid_coarse:
                raise RuntimeError("Grid search found no valid points.")

            g_best_score = float(valid_coarse[0][1].score)
            g_best_theta = valid_coarse[0][3]   # theta echoed in 4th element
            g_best_res = valid_coarse[0][1]
            g_best_names = valid_coarse[0][2]
            best_u_coarse = np.log(np.maximum(g_best_res.f_map, 1e-300) / m)

            print(f"[outer coarse] Best score: {g_best_score:.3f} | "
                  f"{g_best_theta}", flush=True)

            # Stage 2: fine grid around the best coarse point.
            if cfg.outer.two_stage and len(var_names) >= 1:
                fine_n = cfg.outer.fine_grid_points
                if fine_n is None or fine_n <= 0:
                    fine_n = 15
                frac = cfg.outer.fine_grid_factor

                # Build fine bounds centered on coarse best
                fine_bounds = {}
                for k in var_names:
                    orig_lo, orig_hi = cfg.outer.bounds[k]
                    center = g_best_theta[k]
                    if k in ("R_hs", "xi"):
                        # Log-space shrinkage
                        log_c = math.log(max(center, 1e-9))
                        log_span = math.log(max(orig_hi, 1e-9)) - math.log(max(orig_lo, 1e-9))
                        half = frac * log_span * 0.5
                        f_lo = math.exp(max(log_c - half, math.log(max(orig_lo, 1e-9))))
                        f_hi = math.exp(min(log_c + half, math.log(max(orig_hi, 1e-9))))
                    else:
                        span = orig_hi - orig_lo
                        half = frac * span * 0.5
                        f_lo = max(center - half, orig_lo)
                        f_hi = min(center + half, orig_hi)
                    fine_bounds[k] = (f_lo, f_hi)

                print(f"[outer fine] Fine bounds: {fine_bounds}", flush=True)

                u0_fine = best_u_coarse if cfg.outer.fine_grid_warm_start else None
                valid_fine = _run_grid(fine_bounds, fine_n,
                                      stage_label=" fine",
                                      u0_seed=u0_fine)

                if valid_fine:
                    if valid_fine[0][1].score > g_best_score:
                        g_best_score = float(valid_fine[0][1].score)
                        g_best_theta = valid_fine[0][3]   # theta echoed in 4th element
                        g_best_res = valid_fine[0][1]
                        g_best_names = valid_fine[0][2]
                        print(f"[outer fine] Improved score: {g_best_score:.3f} | "
                              f"{g_best_theta}", flush=True)
                    else:
                        print(f"[outer fine] No improvement over coarse best.", flush=True)

                    # Merge top-K from both stages
                    all_valid = valid_coarse + valid_fine
                    all_valid.sort(key=lambda x: x[1].score, reverse=True)
                else:
                    all_valid = valid_coarse
            else:
                all_valid = valid_coarse

            # Local refinement of top-K candidates.
            top_k = all_valid[:cfg.outer.local_opt_top_k]
            print(f"[outer] Refining top {len(top_k)} with Nelder-Mead...", flush=True)

            for rank, (_, res_grid, _, th_start) in enumerate(top_k):
                # theta_sf is echoed directly in the 4th tuple element.
                x_start = np.array([th_start[k] for k in var_names], dtype=float)

                # Warm-start cache from the grid result.
                u_cache = [np.log(np.maximum(res_grid.f_map, 1e-300) / m)]

                def neg_score(x_params, _cache=u_cache):
                    th_eval = {}
                    for j, k in enumerate(var_names):
                        th_eval[k] = float(x_params[j])
                    if not _in_bounds(th_eval, cfg.outer.bounds):
                        return 1e12
                    try:
                        A_tmp, _ = _build_design_matrix(
                            q=q, K=K, sf_model=cfg.sf_model, sf_params=th_eval,
                            include_background=cfg.include_background,
                            include_powerlaw=cfg.include_powerlaw,
                            powerlaw_exponent=cfg.powerlaw_exponent)
                        solver_tmp = MaxEntMAP(A=A_tmp, y=y, sigma=sigma,
                                               m=m, cfg=cfg.maxent)
                        # Profile out alpha and warm-start from nearby solutions.
                        res_tmp = solver_tmp.solve(u0=_cache[0])
                        # Update warm-start cache.
                        _cache[0] = np.log(np.maximum(res_tmp.f_map, 1e-300) / m)

                        I_sc = (res_tmp.I_map_trunc if res_tmp.I_map_trunc is not None
                                else res_tmp.I_map)
                        sc = solver_tmp.laplace_score(
                            res_tmp.f_map, I_sc,
                            res_tmp.alpha, res_tmp.C, res_tmp.S)
                        sc += 0.5 * cfg.n_r * math.log(max(res_tmp.alpha, 1e-300))
                        return -float(sc)
                    except Exception:
                        return 1e12

                bn = [cfg.outer.bounds[k] for k in var_names]

                n_dim = len(var_names)
                init_simplex = np.zeros((n_dim + 1, n_dim), dtype=float)
                init_simplex[0, :] = x_start
                for j, k in enumerate(var_names):
                    lo, hi = bn[j]
                    span = max(float(hi - lo), 1e-12)
                    step = float(cfg.outer.step_scale.get(k, 0.1))
                    if (not np.isfinite(step)) or (step <= 0):
                        step = 0.1

                    if k in ("R_hs", "xi"):
                        base = max(abs(x_start[j]), span * 0.1)
                        delta = step * base
                    else:
                        delta = step

                    delta = min(max(delta, span * 1e-6), span * 0.5)
                    xj = float(np.clip(x_start[j] + delta, lo, hi))
                    if xj == x_start[j]:
                        xj = float(np.clip(x_start[j] - delta, lo, hi))
                    if xj == x_start[j]:
                        eps = max(span * 0.01, 1e-9)
                        xj = float(np.clip(x_start[j] + eps, lo, hi))
                        if xj == x_start[j]:
                            xj = float(np.clip(x_start[j] - eps, lo, hi))

                    init_simplex[j + 1, :] = x_start
                    init_simplex[j + 1, j] = xj

                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    opt_res = minimize(
                        neg_score, x_start, method='Nelder-Mead',
                        bounds=bn,
                        options={'maxiter': 500,
                                 'initial_simplex': init_simplex,
                                 'xatol': cfg.outer.local_opt_xtol,
                                 'fatol': cfg.outer.local_opt_ftol})

                th_opt = {}
                for j, k in enumerate(var_names):
                    th_opt[k] = float(np.clip(opt_res.x[j], bn[j][0], bn[j][1]))

                # Final evaluation with warm-start.
                res_opt, names_opt = evaluate(th_opt, u0=u_cache[0])

                print(f"  Refined #{rank+1}: score={res_opt.score:.3f} | "
                      f"alpha={res_opt.alpha:.3e} | {th_opt}", flush=True)

                if np.isfinite(res_opt.score) and res_opt.score > g_best_score:
                    g_best_score = res_opt.score
                    g_best_theta = th_opt
                    g_best_res = res_opt
                    g_best_names = names_opt

            best_theta = g_best_theta
            best_res = g_best_res
            best_names = g_best_names

    else:
        raise ValueError(f"Unknown outer mode: {cfg.outer.mode}")

    # Final evaluation with warm-start for best theta.
    best_u = np.log(np.maximum(best_res.f_map, 1e-300) / m)
    res_final, names_final = evaluate(best_theta, u0=best_u)
    best_res = res_final

    # Extract result fields.
    x = best_res.f_map
    psd_bins = x[:cfg.n_r].copy()
    extras: Dict[str, float] = {}
    idx = cfg.n_r
    if cfg.include_background:
        extras["background"] = float(x[idx])
        idx += 1
    if cfg.include_powerlaw:
        extras["powerlaw_amp"] = float(x[idx])
        idx += 1

    return PSDInversionResult(
        edges=edges, r_centers=r_centers, psd_bins=psd_bins, extras=extras,
        sf_params=best_theta, alpha=best_res.alpha,
        S=best_res.S, C=best_res.C, score=best_res.score,
        I_fit=best_res.I_map, D=best_res.D,
    )


# ---------------------------------------------------------------------
# Convenience: save PSD CSV
# ---------------------------------------------------------------------
def save_psd_csv(path: str, edges: Array, centers: Array, psd_bins: Array) -> None:
    r_low = edges[:-1]
    r_high = edges[1:]
    w = r_high - r_low
    w_log = np.log(r_high / np.maximum(r_low, 1e-300))
    psd_density = psd_bins / np.maximum(w, 1e-300)
    psd_density_logr = psd_bins / np.maximum(w_log, 1e-300)
    data = np.column_stack([r_low, r_high, centers, psd_bins, psd_density, psd_density_logr])
    header = "r_low,r_high,r_center,psd_bin_weight,psd_density_per_r,psd_density_per_log_r"
    np.savetxt(path, data, delimiter=",", header=header, comments="")


# ---------------------------------------------------------------------
# Entry point: launch GUI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import importlib.util
    _this_file = os.path.abspath(__file__)
    _code_dir = os.path.dirname(_this_file)

    os.environ["MAXENT_CORE_FILE"] = _this_file
    if _code_dir not in sys.path:
        sys.path.insert(0, _code_dir)

    _plot_path = os.path.join(_code_dir, "Maxent_plot_log_v3.2.1.py")
    if not os.path.exists(_plot_path):
        print(f"ERROR: Cannot find {_plot_path}")
        sys.exit(1)
    _spec = importlib.util.spec_from_file_location("Maxent_plot_log", _plot_path)
    _mod = importlib.util.module_from_spec(_spec)
    _mod.__dict__["_DEFAULT_CORE_VERSION"] = "v3.2.1"
    _spec.loader.exec_module(_mod)
    _mod.run_gui_app()

