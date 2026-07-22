import numpy as np

from star_ris_rsma.action import DecodedAction, decode_action, encode_action, project_simplex


def test_project_simplex_feasible():
    projected = project_simplex(np.array([-2.0, 0.2, 3.0]), total=2.0)
    assert np.all(projected >= 0)
    assert np.isclose(projected.sum(), 2.0)


def test_encode_decode_physical_roundtrip():
    action = DecodedAction(
        powers=np.array([0.1, 0.3, 0.6]),
        common_fractions=np.array([0.25, 0.75]),
        beta_t=np.array([0.2, 0.8]),
        theta_t=np.array([-1.0, 1.0]),
        theta_r=np.array([0.5, -0.5]),
    )
    decoded = decode_action(encode_action(action, 1.0), 2, 2, 1.0)
    assert np.allclose(decoded.powers, action.powers, atol=1e-7)
    assert np.allclose(decoded.common_fractions, action.common_fractions, atol=1e-7)
    assert np.allclose(decoded.beta_t, action.beta_t, atol=1e-7)
    assert np.allclose(decoded.theta_t, action.theta_t, atol=1e-6)
