from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .networks import mlp


class PPOAgent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.actor = mlp([obs_dim, hidden_dim, hidden_dim, action_dim]).to(self.device)
        self.critic = mlp([obs_dim, hidden_dim, hidden_dim, 1]).to(self.device)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5, device=self.device))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

    def distribution(self, obs: torch.Tensor) -> Normal:
        return Normal(self.actor(obs), self.log_std.exp())

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.distribution(x)
        pre = dist.mean if deterministic else dist.sample()
        action = torch.tanh(pre)
        log_prob = (dist.log_prob(pre) - torch.log(1.0 - action.pow(2) + 1e-6)).sum(-1)
        value = self.critic(x).squeeze(-1)
        return action.cpu().numpy()[0].astype(np.float32), float(log_prob.item()), float(value.item())

    def checkpoint_state(self) -> dict[str, object]:
        return {"model": self.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_checkpoint_state(self, state: dict[str, object], inference_only: bool = False) -> None:
        self.load_state_dict(state["model"])
        if not inference_only and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])

    def update(self, obs, actions, old_logp, returns, advantages, epochs: int = 10, clip_ratio: float = 0.2) -> dict[str, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device).clamp(-0.999, 0.999)
        old_logp_t = torch.as_tensor(old_logp, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        last = {}
        for _ in range(epochs):
            pre = torch.atanh(act_t)
            dist = self.distribution(obs_t)
            logp = (dist.log_prob(pre) - torch.log(1.0 - act_t.pow(2) + 1e-6)).sum(-1)
            ratio = torch.exp(logp - old_logp_t)
            policy_loss = -torch.minimum(ratio * adv_t, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv_t).mean()
            value = self.critic(obs_t).squeeze(-1)
            value_loss = nn.functional.mse_loss(value, ret_t)
            entropy = dist.entropy().sum(-1).mean()
            loss = policy_loss + 0.5 * value_loss - 0.001 * entropy
            self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.parameters(), 1.0); self.optimizer.step()
            last = {"policy_loss": float(policy_loss.item()), "value_loss": float(value_loss.item())}
        return last
