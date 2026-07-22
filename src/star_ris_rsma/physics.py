from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ChannelSample:
    h_direct: np.ndarray  # [K] complex
    g_br: np.ndarray      # [N] complex
    h_ru: np.ndarray      # [K, N] complex
    user_side: np.ndarray # [K], 0 reflection, 1 transmission


def generate_channel(rng: np.random.Generator, n_users: int, n_ris: int) -> ChannelSample:
    def cn(shape: tuple[int, ...], scale: float) -> np.ndarray:
        return scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.0)

    h_direct = cn((n_users,), 0.35)
    g_br = cn((n_ris,), 1.0)
    h_ru = cn((n_users, n_ris), 0.75)
    user_side = np.arange(n_users) % 2
    return ChannelSample(h_direct=h_direct, g_br=g_br, h_ru=h_ru, user_side=user_side)


def star_coefficients(beta_t: np.ndarray, theta_t: np.ndarray, theta_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta_t = np.clip(beta_t, 0.0, 1.0)
    beta_r = 1.0 - beta_t
    phi_t = np.sqrt(beta_t) * np.exp(1j * theta_t)
    phi_r = np.sqrt(beta_r) * np.exp(1j * theta_r)
    return phi_t, phi_r


def effective_channels(channel: ChannelSample, beta_t: np.ndarray, theta_t: np.ndarray, theta_r: np.ndarray) -> np.ndarray:
    phi_t, phi_r = star_coefficients(beta_t, theta_t, theta_r)
    h_eff = channel.h_direct.astype(np.complex128).copy()
    for k in range(channel.h_direct.shape[0]):
        phi = phi_t if channel.user_side[k] == 1 else phi_r
        h_eff[k] += np.sum(channel.h_ru[k].conj() * phi * channel.g_br)
    return h_eff


def rsma_rates(
    h_eff: np.ndarray,
    powers: np.ndarray,
    common_fractions: np.ndarray,
    noise_power: float,
) -> dict[str, np.ndarray | float]:
    """Compute SISO RSMA rates with one common and K private streams.

    powers[0] is common-stream power, powers[1:] private powers.
    All users decode the common stream while treating private streams as noise.
    Each user then decodes its own private stream after SIC of the common stream.
    """
    gains = np.abs(h_eff) ** 2
    p_common = float(powers[0])
    p_private = np.asarray(powers[1:], dtype=float)
    total_private = float(np.sum(p_private))

    sinr_common = gains * p_common / (gains * total_private + noise_power)
    common_decodable = np.log2(1.0 + sinr_common)
    common_rate = float(np.min(common_decodable))

    interference = np.maximum(total_private - p_private, 0.0)
    sinr_private = gains * p_private / (gains * interference + noise_power)
    private_rates = np.log2(1.0 + sinr_private)

    common_fractions = np.maximum(common_fractions, 0.0)
    denom = float(common_fractions.sum())
    if denom <= 1e-12:
        common_fractions = np.full_like(common_fractions, 1.0 / len(common_fractions))
    else:
        common_fractions = common_fractions / denom
    common_alloc = common_rate * common_fractions
    user_rates = private_rates + common_alloc

    return {
        "sum_rate": float(np.sum(user_rates)),
        "user_rates": user_rates,
        "private_rates": private_rates,
        "common_rate": common_rate,
        "common_alloc": common_alloc,
        "sinr_common": sinr_common,
        "sinr_private": sinr_private,
    }
