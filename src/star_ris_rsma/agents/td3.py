from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn

from .networks import Critic, DeterministicActor


class TD3Agent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
        *,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        policy_delay: int = 2,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        gradient_clip_norm: float = 0.0,
        noise_reference_dim: int = 0,
        critic_loss: str = "mse",
        layer_norm: bool = False,
    ):
        self.device = torch.device(device)
        self.action_dim = int(action_dim)
        self.actor = DeterministicActor(
            obs_dim, action_dim, hidden_dim, layer_norm=layer_norm
        ).to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.q1 = Critic(
            obs_dim, action_dim, hidden_dim, layer_norm=layer_norm
        ).to(self.device)
        self.q2 = Critic(
            obs_dim, action_dim, hidden_dim, layer_norm=layer_norm
        ).to(self.device)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        self.gamma, self.tau = gamma, tau
        self.policy_delay = int(policy_delay)
        self.target_noise = float(target_noise)
        self.noise_clip = float(noise_clip)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.noise_reference_dim = int(noise_reference_dim)
        self.critic_loss = critic_loss
        self.update_count = 0

    def _dimension_noise_scale(self) -> float:
        if self.noise_reference_dim <= 0:
            return 1.0
        return min(1.0, math.sqrt(self.noise_reference_dim / self.action_dim))

    @torch.no_grad()
    def act(self, obs: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(x).cpu().numpy()[0]
        if noise_std > 0:
            effective_std = noise_std * self._dimension_noise_scale()
            action = action + np.random.normal(0.0, effective_std, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def update(
        self,
        batch,
        policy_delay: int | None = None,
        target_noise: float | None = None,
        noise_clip: float | None = None,
    ) -> dict[str, float]:
        obs, action, reward, next_obs, done = [
            torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch
        ]
        delay = self.policy_delay if policy_delay is None else int(policy_delay)
        target_std = self.target_noise if target_noise is None else float(target_noise)
        target_clip = self.noise_clip if noise_clip is None else float(noise_clip)
        target_std *= self._dimension_noise_scale()
        target_clip *= self._dimension_noise_scale()

        with torch.no_grad():
            noise = (torch.randn_like(action) * target_std).clamp(
                -target_clip, target_clip
            )
            next_action = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)
            target_q = torch.minimum(
                self.q1_target(next_obs, next_action),
                self.q2_target(next_obs, next_action),
            )
            y = reward + self.gamma * (1.0 - done) * target_q

        q1 = self.q1(obs, action)
        q2 = self.q2(obs, action)
        if self.critic_loss == "huber":
            q_loss = nn.functional.smooth_l1_loss(q1, y) + nn.functional.smooth_l1_loss(q2, y)
        else:
            q_loss = nn.functional.mse_loss(q1, y) + nn.functional.mse_loss(q2, y)
        self.q_opt.zero_grad()
        q_loss.backward()
        if self.gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()),
                self.gradient_clip_norm,
            )
        self.q_opt.step()

        self.update_count += 1
        actor_loss_value = 0.0
        if self.update_count % delay == 0:
            actor_loss = -self.q1(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            if self.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.gradient_clip_norm
                )
            self.actor_opt.step()
            actor_loss_value = float(actor_loss.item())
            self._soft_update()
        return {
            "critic_loss": float(q_loss.item()),
            "actor_loss": actor_loss_value,
            "effective_target_noise": float(target_std),
        }

    def checkpoint_state(self) -> dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "q_opt": self.q_opt.state_dict(),
            "update_count": self.update_count,
        }

    def load_checkpoint_state(self, state: dict[str, object], inference_only: bool = False) -> None:
        self.actor.load_state_dict(state["actor"])
        if inference_only:
            self.actor_target.load_state_dict(state["actor"])
            return
        self.actor_target.load_state_dict(state["actor_target"])
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.q1_target.load_state_dict(state["q1_target"])
        self.q2_target.load_state_dict(state["q2_target"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.q_opt.load_state_dict(state["q_opt"])
        self.update_count = int(state.get("update_count", 0))

    @torch.no_grad()
    def _soft_update(self) -> None:
        for source, target in [
            (self.actor, self.actor_target),
            (self.q1, self.q1_target),
            (self.q2, self.q2_target),
        ]:
            for p, tp in zip(source.parameters(), target.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)
