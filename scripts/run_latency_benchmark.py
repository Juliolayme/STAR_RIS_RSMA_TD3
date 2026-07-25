from __future__ import annotations

import argparse
from pathlib import Path

from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.latency import benchmark_td3_vs_traditional, write_latency_outputs
from star_ris_rsma.scenario_bank import ScenarioBank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenarios", type=int, default=1000)
    parser.add_argument("--actor-repeats", type=int, default=100)
    parser.add_argument("--decode-repeats", type=int, default=100)
    parser.add_argument("--end-to-end-repeats", type=int, default=20)
    parser.add_argument("--actor-warmup", type=int, default=500)
    parser.add_argument("--solver-warmup-scenarios", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    bank = ScenarioBank.load(args.bank, cfg)
    raw, metadata = benchmark_td3_vs_traditional(
        cfg,
        args.checkpoint,
        bank,
        seed=args.seed,
        scenarios=args.scenarios,
        actor_repeats=args.actor_repeats,
        decode_repeats=args.decode_repeats,
        end_to_end_repeats=args.end_to_end_repeats,
        actor_warmup=args.actor_warmup,
        solver_warmup_scenarios=args.solver_warmup_scenarios,
    )
    write_latency_outputs(raw, metadata, Path(args.output_dir))
    print((Path(args.output_dir) / "LATENCY_SUMMARY.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
