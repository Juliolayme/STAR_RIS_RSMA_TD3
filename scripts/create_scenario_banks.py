from __future__ import annotations

import argparse
import json
from pathlib import Path

from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.scenario_bank import assert_disjoint, generate_bank

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--output-dir", default="artifacts/scenario_banks")
p.add_argument("--train-count", type=int, default=10000)
p.add_argument("--validation-count", type=int, default=1000)
p.add_argument("--test-count", type=int, default=1000)
p.add_argument("--train-seed", type=int, default=11001)
p.add_argument("--validation-seed", type=int, default=22001)
p.add_argument("--test-seed", type=int, default=33001)
a = p.parse_args()

cfg = ExperimentConfig.from_yaml(a.config)
out = Path(a.output_dir)
train = generate_bank(cfg, a.train_count, a.train_seed, "train")
validation = generate_bank(cfg, a.validation_count, a.validation_seed, "validation")
test = generate_bank(cfg, a.test_count, a.test_seed, "test")
assert_disjoint(train, validation, test)
paths = {
    "train": out / f"N{cfg.n_ris}_train.npz",
    "validation": out / f"N{cfg.n_ris}_validation.npz",
    "test": out / f"N{cfg.n_ris}_test.npz",
}
for split, bank in [("train", train), ("validation", validation), ("test", test)]:
    bank.save(paths[split])
manifest = {
    split: {"path": str(path), "checksum": bank.checksum(), "metadata": bank.metadata}
    for split, path, bank in [
        ("train", paths["train"], train),
        ("validation", paths["validation"], validation),
        ("test", paths["test"], test),
    ]
}
out.mkdir(parents=True, exist_ok=True)
(out / f"N{cfg.n_ris}_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
