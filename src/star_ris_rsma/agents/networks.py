from __future__ import annotations

import torch
from torch import nn


def mlp(sizes: list[int], output_activation: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for j in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[j], sizes[j + 1]))
        if j < len(sizes) - 2:
            layers.append(nn.ReLU())
        elif output_activation is not None:
            layers.append(output_activation)
    return nn.Sequential(*layers)


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = mlp([obs_dim, hidden_dim, hidden_dim, action_dim], nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = mlp([obs_dim + action_dim, hidden_dim, hidden_dim, 1])

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))
