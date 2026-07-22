from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DecodedAction:
    powers: np.ndarray
    common_fractions: np.ndarray
    beta_t: np.ndarray
    theta_t: np.ndarray
    theta_r: np.ndarray


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def action_dim(n_users: int, n_ris: int) -> int:
    return (n_users + 1) + n_users + 3 * n_ris


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
