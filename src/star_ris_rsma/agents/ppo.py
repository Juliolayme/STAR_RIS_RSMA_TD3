from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .networks import mlp


class PPOAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        device: str = "cpu",
        *,
        lr: float = 3e-4,
        gradient_clip_norm: float = 1.0,
        layer_norm: bool = False,
        epochs: int = 10,
        minibatch_size: int = 0,
        clip_ratio: float = 0.2,
        entropy_coef: float = 1e-3,
        value_coef: float = 0.5,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.actor = mlp(
            [obs_dim, hidden_dim, hidden_dim, action_dim], layer_norm=layer_norm
        ).to(self.device)
        self.critic = mlp(
            [obs_dim, hidden_dim, hidden_dim, 1], layer_norm=layer_norm
        ).to(self.device)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5, device=self.device))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.epochs = int(epochs)
        self.minibatch_size = int(minibatch_size)
        self.clip_ratio = float(clip_ratio)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)

    def distribution(self, obs: torch.Tensor) -> Normal:
        return Normal(self.actor(obs), self.log_std.exp())

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False, return_pre: bool = False):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.distribution(x)
        pre = dist.mean if deterministic else dist.sample()
        action = torch.tanh(pre)
        log_prob = (dist.log_prob(pre) - torch.log(1.0 - action.pow(2) + 1e-6)).sum(-1)
        value = self.critic(x).squeeze(-1)
        squashed = action.cpu().numpy()[0].astype(np.float32)
        if return_pre:
            # Returning the pre-tanh sample keeps the ratio exact: recovering it
            # with atanh needs a clamp that biases saturated actions.
            return (
                squashed,
                pre.cpu().numpy()[0].astype(np.float32),
                float(log_prob.item()),
                float(value.item()),
            )
        return squashed, float(log_prob.item()), float(value.item())

    def checkpoint_state(self) -> dict[str, object]:
        return {"model": self.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_checkpoint_state(self, state: dict[str, object], inference_only: bool = False) -> None:
        self.load_state_dict(state["model"])
        if not inference_only and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])

    def update(self, obs, actions, old_logp, returns, advantages, pre_actions=None) -> dict[str, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device).clamp(-0.999, 0.999)
        if pre_actions is None:
            pre_t = torch.atanh(act_t)
        else:
            pre_t = torch.as_tensor(pre_actions, dtype=torch.float32, device=self.device)
        old_logp_t = torch.as_tensor(old_logp, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        total = obs_t.shape[0]
        batch = self.minibatch_size if self.minibatch_size > 0 else total
        last: dict[str, float] = {}
        for _ in range(self.epochs):
            order = torch.randperm(total, device=self.device)
            for start in range(0, total, batch):
                index = order[start : start + batch]
                dist = self.distribution(obs_t[index])
                squashed = torch.tanh(pre_t[index])
                logp = (
                    dist.log_prob(pre_t[index])
                    - torch.log(1.0 - squashed.pow(2) + 1e-6)
                ).sum(-1)
                ratio = torch.exp(logp - old_logp_t[index])
                slice_adv = adv_t[index]
                policy_loss = -torch.minimum(
                    ratio * slice_adv,
                    ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * slice_adv,
                ).mean()
                value = self.critic(obs_t[index]).squeeze(-1)
                value_loss = nn.functional.mse_loss(value, ret_t[index])
                entropy = dist.entropy().sum(-1).mean()
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
                last = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                }
        return last
