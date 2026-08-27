"""Guards for the matched-implementation V6 comparison.

The first V6 full run compared a heavily stabilised TD3 against vanilla DDPG
and PPO, and silently reported untrained initial policies whenever no
validation step cleared the feasibility rule. These tests pin the fixes.
"""

from pathlib import Path
import json
import runpy
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch

from star_ris_rsma.agents.ppo import PPOAgent
from star_ris_rsma.checkpoints import build_agent, save_checkpoint
from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.experiment_v2 import constrained_validation_summary
from star_ris_rsma.experiment_v3 import _retain_candidate


ROOT = Path(__file__).resolve().parents[1]
N_VALUES = (16, 32, 64, 96, 128)


def test_v6_configs_give_baselines_the_same_stabilisation_as_td3() -> None:
    for n_ris in N_VALUES:
        cfg = ExperimentConfig.from_yaml(
            ROOT / f"configs/v3/pilot_v6_soft_anchor_n{n_ris}.yaml"
        )
        assert cfg.ddpg_layer_norm is cfg.td3_layer_norm is True
        assert cfg.ddpg_critic_loss == cfg.td3_critic_loss == "huber"
        assert cfg.ddpg_gradient_clip_norm == cfg.td3_gradient_clip_norm == 10.0
        assert cfg.ddpg_actor_lr == cfg.td3_actor_lr
        assert cfg.ddpg_critic_lr == cfg.td3_critic_lr
        assert cfg.ppo_layer_norm is True
        assert cfg.ppo_gradient_clip_norm == 10.0
        assert cfg.ppo_minibatch_size > 0
        assert cfg.retained_candidate_checkpoints >= 1


def test_build_agent_passes_baseline_controls_through() -> None:
    cfg = ExperimentConfig(
        ddpg_actor_lr=1e-4,
        ddpg_critic_lr=2e-4,
        ddpg_gradient_clip_norm=10.0,
        ddpg_critic_loss="huber",
        ddpg_layer_norm=True,
        actor_small_final_init=True,
        ppo_lr=5e-5,
        ppo_gradient_clip_norm=7.0,
        ppo_layer_norm=True,
        ppo_minibatch_size=32,
        ppo_epochs=4,
    )
    ddpg = build_agent("ddpg", 12, 5, cfg, "cpu")
    assert ddpg.gradient_clip_norm == 10.0
    assert ddpg.critic_loss == "huber"
    assert ddpg.actor_opt.param_groups[0]["lr"] == 1e-4
    assert ddpg.q_opt.param_groups[0]["lr"] == 2e-4
    assert any(isinstance(m, torch.nn.LayerNorm) for m in ddpg.actor.net)
    assert any(isinstance(m, torch.nn.LayerNorm) for m in ddpg.q.net)

    ppo = build_agent("ppo", 12, 5, cfg, "cpu")
    assert ppo.optimizer.param_groups[0]["lr"] == 5e-5
    assert ppo.gradient_clip_norm == 7.0
    assert (ppo.minibatch_size, ppo.epochs) == (32, 4)
    assert any(isinstance(m, torch.nn.LayerNorm) for m in ppo.actor)


def test_ppo_keeps_the_exact_pre_tanh_sample() -> None:
    torch.manual_seed(0)
    agent = PPOAgent(6, 4, 32, "cpu")
    with torch.no_grad():
        agent.log_std.fill_(2.0)  # wide policy so tanh saturates often
    obs = np.zeros(6, dtype=np.float32)
    saturated = 0
    for _ in range(200):
        action, pre, _, _ = agent.act(obs, return_pre=True)
        assert np.allclose(np.tanh(pre), action, atol=1e-5)
        if np.abs(action).max() > 0.999:
            saturated += 1
            recovered = np.arctanh(np.clip(action, -0.999, 0.999))
            # The old code reconstructed `pre` this way, which is what biased
            # the importance ratio for saturated actions.
            assert not np.allclose(recovered, pre, atol=1e-3)
    assert saturated > 0


def test_constraint_gap_terms_share_one_scale() -> None:
    frame = pd.DataFrame({
        "reward": [1.0],
        "sum_rate": [1.0],
        "qos_fraction": [0.98],   # one full budget under a 0.99 target
        "all_qos": [0.90],        # one full budget under a 0.95 target
        "violation": [0.02],      # one full budget over a 0.01 tolerance
    })
    cfg = ExperimentConfig(
        validation_qos_fraction_target=0.99,
        validation_all_qos_target=0.95,
        validation_violation_tolerance=0.01,
    )
    summary = constrained_validation_summary(frame, cfg, 0)
    assert summary["normalized_qos_fraction_gap"] == pytest.approx(1.0)
    assert summary["normalized_all_qos_gap"] == pytest.approx(1.0)
    assert summary["normalized_violation_gap"] == pytest.approx(1.0)
    assert summary["constraint_gap"] == pytest.approx(3.0)
    # Raw shortfalls stay available for reporting.
    assert summary["qos_fraction_gap"] == pytest.approx(0.01)
    assert summary["violation_gap"] == pytest.approx(0.01)


class _FakeAgent:
    def checkpoint_state(self) -> dict[str, object]:
        return {
            "actor": {"w": torch.zeros(2)},
            "q1": {"w": torch.zeros(2)},
            "actor_opt": {"state": {}},
        }


def _summary(
    step: int, sum_rate: float, *, violation: float = 0.0, qos: float = 1.0
) -> dict[str, object]:
    return {
        "eval_step": step,
        "mean_sum_rate": sum_rate,
        "mean_violation": violation,
        "mean_qos_fraction": qos,
        "mean_all_qos": qos,
        "mean_reward": sum_rate,
    }


def _retain_all(cfg, steps, tmp_path):
    candidates: list[dict[str, object]] = []
    for step, rate, violation in steps:
        _retain_candidate(
            _FakeAgent(),
            "td3",
            cfg,
            _summary(step, rate, violation=violation),
            step,
            tmp_path,
            candidates,
        )
    return candidates


def test_retention_keeps_every_checkpoint_some_tolerance_could_select(tmp_path: Path) -> None:
    cfg = ExperimentConfig(retained_candidate_checkpoints=4)
    # (step, sum_rate, violation): 2000 is dominated by 1000 - lower rate at a
    # higher violation - so no tolerance could ever select it.
    candidates = _retain_all(
        cfg,
        [(1000, 9.0, 0.001), (2000, 8.0, 0.005), (3000, 12.0, 0.010), (4000, 15.0, 0.030)],
        tmp_path,
    )
    assert [item["eval_step"] for item in candidates] == [1000, 3000, 4000]
    on_disk = sorted(p.name for p in tmp_path.glob("candidate_step*.pt"))
    assert on_disk == [
        "candidate_step1000.pt",
        "candidate_step3000.pt",
        "candidate_step4000.pt",
    ]


def test_retained_set_reproduces_selection_at_any_tolerance(tmp_path: Path) -> None:
    cfg = ExperimentConfig(retained_candidate_checkpoints=4)
    history = [(1000, 9.0, 0.001), (2000, 8.0, 0.005), (3000, 12.0, 0.010), (4000, 15.0, 0.030)]
    candidates = _retain_all(cfg, history, tmp_path)

    def winner(pool, tolerance):
        feasible = [item for item in pool if float(item[2] if isinstance(item, tuple) else item["mean_violation"]) <= tolerance]
        if not feasible:
            return None
        key = (lambda item: item[1]) if isinstance(feasible[0], tuple) else (lambda item: item["mean_sum_rate"])
        return max(feasible, key=key)

    for tolerance in (0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05):
        full = winner(history, tolerance)
        kept = winner(candidates, tolerance)
        if full is None:
            assert kept is None
        else:
            assert kept is not None and int(kept["eval_step"]) == full[0]


def test_candidate_retention_ignores_checkpoints_that_miss_qos(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        retained_candidate_checkpoints=3,
        validation_qos_fraction_target=0.99,
        validation_all_qos_target=0.95,
    )
    candidates: list[dict[str, object]] = []
    _retain_candidate(
        _FakeAgent(), "td3", cfg, _summary(500, 99.0, qos=0.5), 500, tmp_path, candidates
    )
    assert candidates == []
    assert list(tmp_path.glob("candidate_step*.pt")) == []


def test_checkpoint_state_scopes_trim_what_evaluation_never_reads(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    paths = {}
    for scope in ("full", "no_optimizer", "policy"):
        paths[scope] = tmp_path / f"{scope}.pt"
        save_checkpoint(paths[scope], "td3", _FakeAgent(), 1, 0.0, cfg, state_scope=scope)
    loaded = {
        scope: set(torch.load(path, weights_only=False)["agent"])
        for scope, path in paths.items()
    }
    assert loaded["full"] == {"actor", "q1", "actor_opt"}
    assert loaded["no_optimizer"] == {"actor", "q1"}
    # inference_only=True reads state["actor"] and nothing else.
    assert loaded["policy"] == {"actor"}
    assert paths["policy"].stat().st_size < paths["full"].stat().st_size

    with pytest.raises(ValueError, match="state_scope"):
        save_checkpoint(tmp_path / "bad.pt", "td3", _FakeAgent(), 1, 0.0, cfg, state_scope="nope")


def _training_archive(path: Path, best: dict[str, float], initial: dict[str, float]) -> None:
    summary = {
        "checkpoints": {"initial": initial, "best": best, "latest": best},
        "learning_gain_vs_initial": best["sum_rate_mean"] - initial["sum_rate_mean"],
    }
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("run/summary.json", json.dumps(summary))


def test_orchestrator_rejects_a_run_that_never_left_its_initialisation(tmp_path: Path) -> None:
    verify_learning = runpy.run_path(
        ROOT / "scripts" / "orchestrate_kaggle_v6_full.py"
    )["verify_learning"]
    stuck, trained = tmp_path / "stuck.zip", tmp_path / "trained.zip"
    _training_archive(stuck, {"sum_rate_mean": 4.3}, {"sum_rate_mean": 4.3})
    _training_archive(trained, {"sum_rate_mean": 14.3}, {"sum_rate_mean": 4.3})
    with zipfile.ZipFile(stuck) as bundle:
        with pytest.raises(RuntimeError, match="untrained initialisation"):
            verify_learning(bundle, stuck)
    with zipfile.ZipFile(trained) as bundle:
        verify_learning(bundle, trained)
