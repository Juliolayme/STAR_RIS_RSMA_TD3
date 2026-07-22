from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


_EPS = 1e-9


@dataclass(slots=True)
class DecodedAction:
    powers: np.ndarray
    common_fractions: np.ndarray
    beta_t: np.ndarray
    theta_t: np.ndarray
    theta_r: np.ndarray

    @property
    def beta_r(self) -> np.ndarray:
        return 1.0 - self.beta_t

    def copy_with(self, **changes: np.ndarray) -> "DecodedAction":
        return replace(self, **changes)


def softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float) - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def project_simplex(values: np.ndarray, total: float = 1.0) -> np.ndarray:
    """Euclidean projection onto {x >= 0, sum(x) = total}."""
    v = np.asarray(values, dtype=float).reshape(-1)
    if total <= 0:
        raise ValueError("simplex total must be positive")
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - total
    rho_candidates = np.nonzero(u - cssv / (np.arange(v.size) + 1) > 0)[0]
    if rho_candidates.size == 0:
        return np.full_like(v, total / v.size)
    rho = int(rho_candidates[-1])
    theta = cssv[rho] / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    return w * (total / max(float(w.sum()), _EPS))


def wrap_phase(theta: np.ndarray) -> np.ndarray:
    return (np.asarray(theta, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def action_dim(n_users: int, n_ris: int) -> int:
    return (n_users + 1) + n_users + 3 * n_ris


def physical_action_dim(n_users: int, n_ris: int) -> int:
    return action_dim(n_users, n_ris)


def decode_action(raw: np.ndarray, n_users: int, n_ris: int, p_max: float) -> DecodedAction:
    raw = np.asarray(raw, dtype=float).reshape(-1)
    expected = action_dim(n_users, n_ris)
    if raw.size != expected:
        raise ValueError(f"Expected action dimension {expected}, got {raw.size}")

    i = 0
    p_logits = raw[i:i + n_users + 1]; i += n_users + 1
    c_logits = raw[i:i + n_users]; i += n_users
    beta_raw = raw[i:i + n_ris]; i += n_ris
    theta_t_raw = raw[i:i + n_ris]; i += n_ris
    theta_r_raw = raw[i:i + n_ris]

    powers = p_max * softmax(p_logits)
    common_fractions = softmax(c_logits)
    beta_t = 1.0 / (1.0 + np.exp(-np.clip(beta_raw, -30.0, 30.0)))
    theta_t = np.pi * np.tanh(theta_t_raw)
    theta_r = np.pi * np.tanh(theta_r_raw)
    return DecodedAction(powers, common_fractions, beta_t, theta_t, theta_r)


def encode_action(action: DecodedAction, p_max: float) -> np.ndarray:
    """Map a feasible physical action back to the unconstrained actor space."""
    powers = project_simplex(action.powers, p_max)
    fractions = project_simplex(action.common_fractions, 1.0)
    beta = np.clip(action.beta_t, 1e-7, 1.0 - 1e-7)
    theta_t = np.clip(wrap_phase(action.theta_t) / np.pi, -0.999999, 0.999999)
    theta_r = np.clip(wrap_phase(action.theta_r) / np.pi, -0.999999, 0.999999)

    p_logits = np.log(np.clip(powers / p_max, _EPS, None))
    p_logits -= p_logits.mean()
    c_logits = np.log(np.clip(fractions, _EPS, None))
    c_logits -= c_logits.mean()
    beta_logits = np.log(beta / (1.0 - beta))
    return np.concatenate([
        p_logits,
        c_logits,
        beta_logits,
        np.arctanh(theta_t),
        np.arctanh(theta_r),
    ]).astype(np.float64)


def flatten_physical(action: DecodedAction) -> np.ndarray:
    return np.concatenate([
        np.asarray(action.powers, dtype=float),
        np.asarray(action.common_fractions, dtype=float),
        np.asarray(action.beta_t, dtype=float),
        wrap_phase(action.theta_t),
        wrap_phase(action.theta_r),
    ])


def unflatten_physical(vector: np.ndarray, n_users: int, n_ris: int, p_max: float) -> DecodedAction:
    x = np.asarray(vector, dtype=float).reshape(-1)
    if x.size != physical_action_dim(n_users, n_ris):
        raise ValueError(f"Expected physical action dimension {physical_action_dim(n_users, n_ris)}, got {x.size}")
    i = 0
    powers = project_simplex(x[i:i+n_users+1], p_max); i += n_users + 1
    fractions = project_simplex(x[i:i+n_users], 1.0); i += n_users
    beta = np.clip(x[i:i+n_ris], 0.0, 1.0); i += n_ris
    theta_t = wrap_phase(x[i:i+n_ris]); i += n_ris
    theta_r = wrap_phase(x[i:i+n_ris])
    return DecodedAction(powers, fractions, beta, theta_t, theta_r)
