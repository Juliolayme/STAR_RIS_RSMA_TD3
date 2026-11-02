"""Kiểm tra lại toàn bộ bằng chứng V6 r2 trong GitHub Release so với số liệu của luận văn.

Các bước (không huấn luyện lại):
1. SHA-256 của từng tệp release khớp EVIDENCE_MANIFEST.json.
2. SHA-256 của 15 ScenarioBank và 75 lưu trữ huấn luyện lồng bên trong khớp manifest;
   checksum nội dung của ngân hàng kiểm thử khớp cột bank_checksum của dữ liệu thô.
3. Mỗi lưu trữ có đủ tệp bắt buộc; best.pt của hạt giống 0 khớp LATENCY_CHECKPOINT_INDEX.json.
4. test_best_raw.csv trong lưu trữ khớp từng dòng results/physical_v6_full_r2/raw/DRL_V6_TEST_BEST_RAW_ALL.csv.
5. Nạp lại best.pt, đánh giá xác định trên ngân hàng kiểm thử đã khóa (CPU) và so từng kịch bản
   với dữ liệu thô; trung bình theo (thuật toán, N) so với TABLE_V6_SIX_METHOD_PERFORMANCE.csv.

Chạy:
  python results/physical_v6_full_r2/evidence/download_evidence.py --destination artifacts/v6_evidence
  python scripts/verify_v6_release_evidence.py --assets artifacts/v6_evidence
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from star_ris_rsma.checkpoints import load_checkpoint  # noqa: E402
from star_ris_rsma.config import ExperimentConfig  # noqa: E402
from star_ris_rsma.env import StarRisRsmaEnv  # noqa: E402
from star_ris_rsma.experiment import evaluate_policy_on_bank  # noqa: E402
from star_ris_rsma.scenario_bank import ScenarioBank  # noqa: E402

V6 = ROOT / "results" / "physical_v6_full_r2"
EVIDENCE = V6 / "evidence"
METHODS = ("td3", "ddpg", "ppo")
N_VALUES = (16, 32, 64, 96, 128)
REQUIRED = {
    "initial.pt", "best.pt", "latest.pt", "summary.json", "timing.json",
    "test_initial_raw.csv", "test_best_raw.csv", "test_latest_raw.csv", "manifest.json",
    "training.csv", "validation_summary.csv", "candidate_checkpoints.json",
    "SCENARIO_BANK_VERIFICATION.json", "KAGGLE_JOB_PROVENANCE.json",
}
TOL = 1e-6


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def member(z: zipfile.ZipFile, name: str) -> str:
    # Một số lưu trữ được tạo trên Windows nên dùng dấu "\\" làm phân cách thư mục.
    hits = [m for m in z.namelist() if m.replace("\\", "/").split("/")[-1] == name]
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name}, got {hits}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=EVIDENCE / "RELEASE_VERIFICATION.json")
    ap.add_argument("--skip-reevaluation", action="store_true")
    args = ap.parse_args()

    manifest = json.loads((EVIDENCE / "EVIDENCE_MANIFEST.json").read_text(encoding="utf-8"))
    index = json.loads((V6 / "LATENCY_CHECKPOINT_INDEX.json").read_text(encoding="utf-8"))
    seed0 = {(c["method"], int(c["n_ris"])): c["checkpoint_sha256"] for c in index["checkpoints"]}
    raw = pd.read_csv(V6 / "raw" / "DRL_V6_TEST_BEST_RAW_ALL.csv")
    perf = pd.read_csv(V6 / "tables" / "TABLE_V6_SIX_METHOD_PERFORMANCE.csv").set_index(["method", "n_ris"])
    report: dict = {"release_tag": manifest["release_tag"], "checks": {}, "failures": []}

    def fail(msg: str) -> None:
        report["failures"].append(msg)
        print("FAIL:", msg, flush=True)

    # 1. top-level assets
    assets = {}
    for key, item in manifest["assets"].items():
        path = args.assets / item["file"]
        digest = sha_file(path)
        assets[key] = path
        ok = digest == item["sha256"]
        report["checks"][f"asset:{item['file']}"] = {"sha256": digest, "match": ok}
        if not ok:
            fail(f"asset sha mismatch {item['file']}")
    print("assets checked", flush=True)

    # 2. scenario banks
    tmp = Path(tempfile.mkdtemp())
    bank_sha = {b["file"]: b["sha256"] for b in manifest["scenario_banks"]}
    test_banks: dict[int, ScenarioBank] = {}
    with zipfile.ZipFile(assets["scenario_banks"]) as z:
        for name, expected in bank_sha.items():
            data = z.read(member(z, name))
            if sha_bytes(data) != expected:
                fail(f"bank sha mismatch {name}")
            if name.endswith("_test.npz"):
                n = int(name[1:].split("_")[0])
                p = tmp / name
                p.write_bytes(data)
                test_banks[n] = ScenarioBank.load(p)
    for n, bank in test_banks.items():
        expected = set(raw.loc[raw.n_ris == n, "bank_checksum"].astype(str))
        ok = expected == {bank.checksum()}
        report["checks"][f"test_bank:N{n}"] = {"checksum": bank.checksum(), "match_raw": ok}
        if not ok:
            fail(f"test bank checksum N{n}")
    print("banks checked", flush=True)

    # 3-5. training archives
    evaluated = []
    for method in METHODS:
        expected = {a["file"]: a for a in manifest["training"][method]["archives"]}
        with zipfile.ZipFile(assets[f"{method}_training"]) as outer:
            for fname, meta in sorted(expected.items()):
                n, seed = int(meta["n_ris"]), int(meta["seed"])
                data = outer.read(member(outer, fname))
                rec = {"method": method, "n_ris": n, "seed": seed, "archive": fname,
                       "archive_sha_match": sha_bytes(data) == meta["sha256"]}
                if not rec["archive_sha_match"]:
                    fail(f"archive sha {fname}")
                with zipfile.ZipFile(io.BytesIO(data)) as inner:
                    names = {m.replace("\\", "/").split("/")[-1] for m in inner.namelist()}
                    missing = sorted(REQUIRED - names)
                    rec["missing_files"] = missing
                    if missing:
                        fail(f"{fname} missing {missing}")
                    best = inner.read(member(inner, "best.pt"))
                    rec["best_pt_sha256"] = sha_bytes(best)
                    if seed == 0:
                        rec["best_pt_matches_latency_index"] = rec["best_pt_sha256"] == seed0[(method, n)]
                        if not rec["best_pt_matches_latency_index"]:
                            fail(f"best.pt seed0 sha {method} N{n}")
                    archived = pd.read_csv(io.BytesIO(inner.read(member(inner, "test_best_raw.csv"))))
                ref = raw[(raw.method == method) & (raw.n_ris == n) & (raw.seed == seed)].sort_values("scenario")
                archived = archived.sort_values("scenario")
                rec["archived_vs_raw_max_abs_sum_rate"] = float(
                    np.max(np.abs(archived.sum_rate.to_numpy() - ref.sum_rate.to_numpy())))
                rec["archived_vs_raw_all_qos_equal"] = bool(
                    (archived.all_qos.astype(bool).to_numpy() == ref.all_qos.astype(bool).to_numpy()).all())
                if len(archived) != 1000 or rec["archived_vs_raw_max_abs_sum_rate"] > TOL or not rec["archived_vs_raw_all_qos_equal"]:
                    fail(f"archived test csv differs {fname}")

                if not args.skip_reevaluation:
                    cfg = ExperimentConfig.from_yaml(ROOT / f"configs/v3/pilot_v6_soft_anchor_n{n}.yaml")
                    env = StarRisRsmaEnv(cfg, seed)
                    pt = tmp / "best.pt"
                    pt.write_bytes(best)
                    agent, _ = load_checkpoint(pt, method, env.observation_dim, env.action_dim, cfg, "cpu")
                    rows = evaluate_policy_on_bank(agent, method, cfg, test_banks[n], seed).sort_values("scenario")
                    diff = np.abs(rows.sum_rate.to_numpy() - ref.sum_rate.to_numpy())
                    rec["reeval_max_abs_sum_rate"] = float(diff.max())
                    rec["reeval_mean_sum_rate"] = float(rows.sum_rate.mean())
                    rec["reeval_all_qos_mean"] = float(rows.all_qos.astype(float).mean())
                    rec["reeval_all_qos_flips"] = int(
                        (rows.all_qos.astype(bool).to_numpy() != ref.all_qos.astype(bool).to_numpy()).sum())
                evaluated.append(rec)
                print(method, n, seed, {k: v for k, v in rec.items() if k.startswith(("archive_sha", "reeval_max", "archived_vs"))}, flush=True)

    report["archives"] = evaluated
    if not args.skip_reevaluation:
        df = pd.DataFrame(evaluated)
        agg = df.groupby(["method", "n_ris"]).agg(sum_rate=("reeval_mean_sum_rate", "mean"),
                                                  sum_rate_std=("reeval_mean_sum_rate", "std"),
                                                  all_qos=("reeval_all_qos_mean", "mean")).reset_index()
        table = []
        for r in agg.itertuples():
            p = perf.loc[(r.method, r.n_ris)]
            row = {"method": r.method, "n_ris": int(r.n_ris),
                   "reeval_sum_rate_mean": r.sum_rate, "thesis_sum_rate_mean": float(p.sum_rate_mean),
                   "reeval_sum_rate_std": r.sum_rate_std, "thesis_sum_rate_std": float(p.sum_rate_std),
                   "reeval_all_qos_mean": r.all_qos, "thesis_all_qos_mean": float(p.all_qos_mean)}
            row["abs_diff_sum_rate_mean"] = abs(row["reeval_sum_rate_mean"] - row["thesis_sum_rate_mean"])
            table.append(row)
        report["table_comparison"] = table
        report["max_reeval_scenario_abs_diff"] = float(df.reeval_max_abs_sum_rate.max())
        report["max_table_abs_diff_sum_rate_mean"] = float(max(r["abs_diff_sum_rate_mean"] for r in table))
        report["total_all_qos_flips"] = int(df.reeval_all_qos_flips.sum())
    report["n_archives"] = len(evaluated)
    report["result"] = "PASS" if not report["failures"] else "FAIL"
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("RESULT", report["result"], "failures", len(report["failures"]))


if __name__ == "__main__":
    main()
