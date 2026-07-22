from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..action import DecodedAction, flatten_physical, project_simplex, unflatten_physical, wrap_phase
from ..env import StarRisRsmaEnv


@dataclass(slots=True)
class SolverState:
    vector: np.ndarray
    action: DecodedAction
    metrics: dict[str, object]

    @property
    def score(self) -> float:
        return float(self.metrics["reward"])


def state_from_action(env: StarRisRsmaEnv, action: DecodedAction) -> SolverState:
    feasible = unflatten_physical(
        flatten_physical(action), env.config.n_users, env.config.n_ris, env.config.p_max
    )
    return SolverState(flatten_physical(feasible), feasible, env.evaluate_decoded_action(feasible))


def state_from_vector(env: StarRisRsmaEnv, vector: np.ndarray) -> SolverState:
    action = unflatten_physical(vector, env.config.n_users, env.config.n_ris, env.config.p_max)
    return SolverState(flatten_physical(action), action, env.evaluate_decoded_action(action))


def physical_slices(n_users: int, n_ris: int) -> dict[str, slice]:
    i = 0
    slices = {"powers": slice(i, i + n_users + 1)}; i += n_users + 1
    slices["common"] = slice(i, i + n_users); i += n_users
    slices["beta"] = slice(i, i + n_ris); i += n_ris
    slices["theta_t"] = slice(i, i + n_ris); i += n_ris
    slices["theta_r"] = slice(i, i + n_ris)
    return slices


def project_physical(vector: np.ndarray, n_users: int, n_ris: int, p_max: float) -> np.ndarray:
    x = np.asarray(vector, dtype=float).copy()
    s = physical_slices(n_users, n_ris)
    x[s["powers"]] = project_simplex(x[s["powers"]], p_max)
    x[s["common"]] = project_simplex(x[s["common"]], 1.0)
    x[s["beta"]] = np.clip(x[s["beta"]], 0.0, 1.0)
    x[s["theta_t"]] = wrap_phase(x[s["theta_t"]])
    x[s["theta_r"]] = wrap_phase(x[s["theta_r"]])
    return x


def merit(metrics: dict[str, object]) -> float:
    return float(metrics["reward"])
