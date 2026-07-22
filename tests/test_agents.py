import numpy as np

from star_ris_rsma.agents import DDPGAgent, PPOAgent, TD3Agent


def test_deterministic_eval_actions():
    obs = np.zeros(7, dtype=np.float32)
    for cls in [TD3Agent, DDPGAgent]:
        agent = cls(7, 5, 16)
        assert np.allclose(agent.act(obs, 0.0), agent.act(obs, 0.0))


def test_ppo_deterministic_eval():
    agent = PPOAgent(7, 5, 16)
    obs = np.zeros(7, dtype=np.float32)
    a1, _, _ = agent.act(obs, deterministic=True)
    a2, _, _ = agent.act(obs, deterministic=True)
    assert np.allclose(a1, a2)
