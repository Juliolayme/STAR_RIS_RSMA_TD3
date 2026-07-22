from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from .networks import Critic, DeterministicActor


class DDPGAgent:
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, gamma: float = 0.99, tau: float = 0.005, device: str = "cpu"):
        self.device = torch.device(device)
        self.actor = DeterministicActor(obs_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.q = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.q_target = deepcopy(self.q)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=3e-4)
        self.gamma, self.tau = gamma, tau

    @torch.no_grad()
    def act(self, obs: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(x).cpu().numpy()[0]
        if noise_std > 0:
            action += np.random.normal(0.0, noise_std, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def update(self, batch) -> dict[str, float]:
        obs, action, reward, next_obs, done = [torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch]
        with torch.no_grad():
            y = reward + self.gamma * (1.0 - done) * self.q_target(next_obs, self.actor_target(next_obs))
        q_loss = nn.functional.mse_loss(self.q(obs, action), y)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()
        actor_loss = -self.q(obs, self.actor(obs)).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
        self._soft_update()
        return {"critic_loss": float(q_loss.item()), "actor_loss": float(actor_loss.item())}

    @torch.no_grad()
    def _soft_update(self) -> None:
        for source, target in [(self.actor, self.actor_target), (self.q, self.q_target)]:
            for p, tp in zip(source.parameters(), target.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)
