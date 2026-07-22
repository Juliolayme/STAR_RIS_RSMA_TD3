# Method implementation contract

## Primary learned method and DRL baselines

- TD3 is the primary DRL method.
- DDPG and PPO are DRL baselines.
- All learned methods use the same observation, action decoder, reward, locked ScenarioBank protocol and deterministic test evaluation.
- Best checkpoints are selected only by mean validation reward. The test bank is never used for checkpoint selection.

## AO-SCA

The implementation is a two-block **proximal first-order AO-SCA** on feasible physical variables:

1. RSMA block: stream powers and common-rate fractions.
2. STAR-RIS block: energy-splitting coefficients and transmit/reflect phases.

At point \(x^{(t)}\), each block uses the surrogate

\[
\widetilde f(z;x^{(t)}) =
\nabla f(x^{(t)})^T(z-x^{(t)})
-\frac{\rho}{2}\|z-x^{(t)}\|_2^2.
\]

The constrained surrogate maximizer is the projection of
\(x^{(t)}+\nabla f(x^{(t)})/\rho\) onto:

- the total-power simplex;
- the common-rate-fraction simplex;
- \([0,1]^N\) for energy splitting;
- the periodic phase domain \([-\pi,\pi)^N\).

Finite differences estimate the first-order gradient. Backtracking increases
\(\rho\) until both surrogate gain and exact merit gain are non-negative.
The algorithm is local and initialization-dependent. It is not a global optimum,
upper bound, or proof of optimality.

## AO-Grid

AO-Grid is a deterministic alternating coordinate codebook search:

- power and common-rate coordinates use finite allocation grids while residual mass is redistributed to remain on the simplex;
- energy splitting uses a fixed beta grid;
- transmit and reflect phases use fixed phase codebooks;
- every declared candidate is explicitly evaluated and the best non-decreasing candidate is retained.

It does not use random perturbations and is separate from AO-SCA.

## AnalyticalRIS

AnalyticalRIS aligns a shared phase vector to the aggregate cascaded channel and uses:

- \(\beta_n^T=\beta_n^R=0.5\);
- equal power over the common and private streams;
- equal common-rate fractions.

Therefore it must be reported as **analytical phase alignment with equal allocation**, not as a complete analytical optimizer of every resource variable.

## Locked evaluation

Scenario banks are saved as NPZ files with metadata and SHA-256 checksums.
Train, validation and test banks use different seeds and are checked for duplicate channel realizations.
Every result manifest records the configuration hash, bank checksum and Git commit.
