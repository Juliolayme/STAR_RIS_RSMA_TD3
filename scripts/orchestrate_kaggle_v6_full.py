from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import time
import zipfile


METHODS = ("td3", "ddpg", "ppo")
# Bumped whenever the training protocol changes. It is part of every kernel
# slug so a rerun cannot adopt a finished kernel from an earlier protocol and
# silently mix its results into the new manifest.
PROTOCOL_REVISION = "r2"
N_VALUES = (16, 32, 64, 96, 128)
SEEDS = tuple(range(5))
MAX_ACTIVE_PER_ACCOUNT = 2
ASSIGNMENTS = {
    "td3": (
        (16, 0), (16, 3), (32, 1), (32, 4), (64, 2),
        (96, 0), (96, 3), (128, 1), (128, 4),
    ),
    "ddpg": (
        (16, 1), (16, 4), (32, 2), (64, 0),
        (64, 3), (96, 1), (96, 4), (128, 2),
    ),
    "ppo": (
        (16, 2), (32, 0), (32, 3), (64, 1),
        (64, 4), (96, 2), (128, 0), (128, 3),
    ),
}
USERNAMES = {
    "td3": "thanhnguyen1423",
    "ddpg": "ronganminh",
    "ppo": "duythanhb1909984",
}
KEY_ENV = {
    "td3": "KAGGLE_TD3_KEY",
    "ddpg": "KAGGLE_DDPG_KEY",
    "ppo": "KAGGLE_PPO_KEY",
}
REQUIRED_ARCHIVE_FILES = {
    "initial.pt",
    "best.pt",
    "latest.pt",
    "summary.json",
    "timing.json",
    "test_initial_raw.csv",
    "test_best_raw.csv",
    "test_latest_raw.csv",
    "manifest.json",
    "training.csv",
    "validation_summary.csv",
    "candidate_checkpoints.json",
    "SCENARIO_BANK_VERIFICATION.json",
    "KAGGLE_JOB_PROVENANCE.json",
}


def slugify(title: str) -> str:
    """Kaggle's own slug derivation, reproduced so titles and ids cannot drift."""
    lowered = "".join(
        character if character.isalnum() else "-" for character in title.lower()
    )
    while "--" in lowered:
        lowered = lowered.replace("--", "-")
    return lowered.strip("-")


@dataclass(slots=True)
class Job:
    account: str
    n_ris: int
    seed: int
    method: str = "td3"
    revision: str = PROTOCOL_REVISION
    attempts: int = 0
    status: str = "pending"
    status_text: str = ""
    archive_sha256: str | None = None
    archive_bytes: int | None = None

    @property
    def name_stem(self) -> str:
        return f"physical_v6_{self.method}_{self.revision}"

    @property
    def slug(self) -> str:
        return (
            f"star-ris-{self.method}-v6-full-{self.revision}"
            f"-n{self.n_ris}-seed-{self.seed}"
        )

    @property
    def title(self) -> str:
        # Kaggle derives a kernel's slug from its title and rejects a push
        # whose id disagrees, so the revision has to appear in both.
        return (
            f"STAR-RIS {self.method.upper()} V6 full {self.revision} "
            f"N{self.n_ris} seed {self.seed}"
        )

    @property
    def ref(self) -> str:
        return f"{USERNAMES[self.account]}/{self.slug}"

    @property
    def tag(self) -> str:
        return f"{self.name_stem}_n{self.n_ris}_100k"

    @property
    def output_root(self) -> str:
        return f"/kaggle/working/{self.name_stem}_full"

    @property
    def archive_name(self) -> str:
        return f"{self.name_stem}_n{self.n_ris}_seed{self.seed}.zip"

    def record(self) -> dict[str, object]:
        return {
            "account": self.account,
            "username": USERNAMES[self.account],
            "method": self.method,
            "protocol_revision": self.revision,
            "n_ris": self.n_ris,
            "seed": self.seed,
            "ref": self.ref,
            "attempts": self.attempts,
            "status": self.status,
            "status_text": self.status_text,
            "archive": self.archive_name,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
        }


def credential_env(account: str) -> dict[str, str]:
    key = os.environ.get(KEY_ENV[account], "")
    if not key:
        raise RuntimeError(f"Missing secret environment variable {KEY_ENV[account]}")
    return {
        **os.environ,
        "KAGGLE_USERNAME": USERNAMES[account],
        "KAGGLE_KEY": key,
    }


def run_cli(account: str, arguments: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["kaggle", *arguments],
        env=credential_env(account),
        text=True,
        capture_output=True,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if output:
        print(output, flush=True)
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Kaggle CLI failed for {account}: kaggle {' '.join(arguments)}"
        )
    return completed


def classify_status(job: Job) -> tuple[str, str]:
    completed = run_cli(job.account, ["kernels", "status", job.ref])
    text = f"{completed.stdout}\n{completed.stderr}".strip()
    normalized = text.lower()
    if completed.returncode != 0 and any(
        token in normalized for token in ("404", "not found", "does not exist")
    ):
        return "missing", text
    if any(token in normalized for token in ("complete", "success")):
        return "complete", text
    if any(token in normalized for token in ("error", "failed", "cancel")):
        return "failed", text
    if "running" in normalized:
        return "running", text
    if any(token in normalized for token in ("queued", "pending")):
        return "queued", text
    if completed.returncode != 0:
        return "missing", text
    return "unknown", text


def runner_source(job: Job, commit: str) -> str:
    tag = job.tag
    return textwrap.dedent(
        f'''\
        from __future__ import annotations
        import json
        import shutil
        import subprocess
        import sys
        from datetime import datetime, timezone
        from pathlib import Path

        repo = Path("/kaggle/working/STAR_RIS_RSMA_TD3")
        output_root = Path("{job.output_root}")
        if repo.exists():
            shutil.rmtree(repo)
        subprocess.run(["git", "clone", "--filter=blob:none", "https://github.com/Juliolayme/STAR_RIS_RSMA_TD3.git", str(repo)], check=True)
        subprocess.run(["git", "checkout", "--detach", "{commit}"], cwd=repo, check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "-e", str(repo)], check=True)
        subprocess.run([
            sys.executable, "scripts/create_scenario_banks.py",
            "--config", "configs/v3/pilot_v6_soft_anchor_n{job.n_ris}.yaml",
            "--output-dir", "artifacts/scenario_banks",
            "--train-count", "10000", "--validation-count", "1000", "--test-count", "1000",
        ], cwd=repo, check=True)
        verification = Path("/kaggle/working/scenario_bank_verification_n{job.n_ris}.json")
        subprocess.run([
            sys.executable, "scripts/prepare_v6_scenario_banks.py",
            "--n-ris", "{job.n_ris}", "--verify-existing", "--manifest", str(verification),
        ], cwd=repo, check=True)
        subprocess.run([
            sys.executable, "scripts/pilot_structure_aware_td3.py",
            "--method", "{job.method}",
            "--config", "configs/v3/pilot_v6_soft_anchor_n{job.n_ris}.yaml",
            "--seed", "{job.seed}", "--tag", "{tag}",
            "--output-root", str(output_root),
        ], cwd=repo, check=True)
        run_dir = output_root / "{tag}_seed{job.seed}"
        shutil.copy2(verification, run_dir / "SCENARIO_BANK_VERIFICATION.json")
        provenance = {{
            "experiment": "physical_v6_soft_anchor_full",
            "method": "{job.method}",
            "n_ris": {job.n_ris},
            "seed": {job.seed},
            "train_steps": 100000,
            "repository": "Juliolayme/STAR_RIS_RSMA_TD3",
            "git_commit": "{commit}",
            "kaggle_account_label": "{job.account}",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }}
        (run_dir / "KAGGLE_JOB_PROVENANCE.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )
        shutil.make_archive(
            "/kaggle/working/{job.archive_name.removesuffix('.zip')}",
            "zip", output_root, run_dir.name,
        )
        '''
    )


def build_submission(job: Job, commit: str, root: Path) -> Path:
    target = root / f"{job.method}_{job.account}_n{job.n_ris}_seed{job.seed}"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    if slugify(job.title) != job.slug:
        raise RuntimeError(
            f"Kaggle would derive slug {slugify(job.title)!r} from title "
            f"{job.title!r}, which does not match the id slug {job.slug!r}"
        )
    (target / "runner.py").write_text(runner_source(job, commit), encoding="utf-8")
    metadata = {
        "id": job.ref,
        "title": job.title,
        "code_file": "runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "enable_tpu": False,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
    }
    (target / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return target


def submit(job: Job, commit: str, submission_root: Path) -> None:
    target = build_submission(job, commit, submission_root)
    job.attempts += 1
    job.status = "submitting"
    print(f"SUBMIT {job.ref} attempt={job.attempts}", flush=True)
    run_cli(
        job.account,
        [
            "kernels", "push", "-p", str(target),
            "--accelerator", "NvidiaTeslaT4", "--timeout", "43200",
        ],
        check=True,
    )
    job.status = "queued"


class JobVerificationError(RuntimeError):
    """A kernel finished but its archive is not acceptable evidence.

    Raised instead of a bare RuntimeError so one unusable archive costs that
    job a retry rather than aborting the orchestrator and discarding the
    other twenty-four results along with the manifest.
    """


def verify_learning(bundle: zipfile.ZipFile, archive: Path) -> None:
    """Reject a run whose selected checkpoint is the untrained initialisation.

    A run can finish cleanly and still hand back its initial policy as `best`
    when no validation step ever clears the feasibility rule. That reads as a
    completed job everywhere else, so it is checked here.
    """
    matches = [
        name for name in bundle.namelist() if name.endswith("/summary.json")
    ]
    if len(matches) != 1:
        raise JobVerificationError(f"{archive}: expected one summary.json, got {matches}")
    summary = json.loads(bundle.read(matches[0]))
    checkpoints = summary["checkpoints"]
    if checkpoints["best"] == checkpoints["initial"]:
        raise JobVerificationError(
            f"{archive}: best checkpoint equals the untrained initialisation "
            "- no validation step was ever selected"
        )
    if float(summary["learning_gain_vs_initial"]) <= 0.0:
        raise JobVerificationError(
            f"{archive}: learning_gain_vs_initial is "
            f"{summary['learning_gain_vs_initial']}, expected a positive gain"
        )


def verify_download(job: Job, output_dir: Path) -> None:
    archive = output_dir / job.archive_name
    if not archive.is_file() or archive.stat().st_size == 0:
        raise JobVerificationError(f"Missing output archive {archive}")
    with zipfile.ZipFile(archive) as bundle:
        names = {
            Path(name).name
            for name in bundle.namelist()
            if not name.endswith("/")
        }
        missing = sorted(REQUIRED_ARCHIVE_FILES - names)
        if missing:
            raise JobVerificationError(f"{archive} is missing required files: {missing}")
        verify_learning(bundle, archive)
    job.archive_bytes = archive.stat().st_size
    job.archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    job.status = "complete"


def download(job: Job, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_cli(
        job.account,
        ["kernels", "output", job.ref, "-p", str(output_dir)],
        check=True,
    )
    verify_download(job, output_dir)


def write_manifest(path: Path, jobs: list[Job], commit: str) -> None:
    completed = [job for job in jobs if job.status == "complete"]
    failed = [job for job in jobs if job.status == "failed"]
    manifest = {
        "audit": (
            "PASS" if len(completed) == 25 and not failed else
            "FAIL" if failed else "RUNNING"
        ),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": "Juliolayme/STAR_RIS_RSMA_TD3",
        "git_commit": commit,
        "protocol": {
            "method": jobs[0].method,
            "revision": jobs[0].revision,
            "action_parameterization": "physical_v6_soft_anchor",
            "n_values": list(N_VALUES),
            "seeds": list(SEEDS),
            "train_steps": 100_000,
            "expected_jobs": 25,
            "max_active_per_account": MAX_ACTIVE_PER_ACCOUNT,
            "max_active_total": 6,
        },
        "completed_jobs": len(completed),
        "failed_jobs": len(failed),
        "jobs": [job.record() for job in jobs],
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--method", choices=METHODS, default="td3")
    parser.add_argument("--revision", default=PROTOCOL_REVISION)
    parser.add_argument("--output-dir", type=Path, default=Path("collected"))
    parser.add_argument("--submission-root", type=Path, default=Path("kaggle_submit_v6"))
    parser.add_argument("--manifest", type=Path, default=Path("TRAINING_RUN_MANIFEST.json"))
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-minutes", type=int, default=210)
    args = parser.parse_args()

    jobs = [
        Job(account, n_ris, seed, args.method, args.revision)
        for account, combinations in ASSIGNMENTS.items()
        for n_ris, seed in combinations
    ]
    expected = {(n_ris, seed) for n_ris in N_VALUES for seed in SEEDS}
    observed = {(job.n_ris, job.seed) for job in jobs}
    if len(jobs) != 25 or observed != expected:
        raise RuntimeError("ASSIGNMENTS must contain all 25 unique N/seed jobs")

    pending = {
        account: deque(job for job in jobs if job.account == account)
        for account in ASSIGNMENTS
    }
    active: dict[str, list[Job]] = {account: [] for account in ASSIGNMENTS}
    deadline = time.monotonic() + args.timeout_minutes * 60

    def fill(account: str) -> None:
        while (
            len(active[account]) < MAX_ACTIVE_PER_ACCOUNT
            and pending[account]
        ):
            job = pending[account].popleft()
            state, status_text = classify_status(job)
            job.status_text = status_text
            if state == "complete":
                print(f"ADOPT COMPLETE {job.ref}", flush=True)
                try:
                    download(job, args.output_dir)
                except JobVerificationError as error:
                    print(f"REJECT ADOPTED {job.ref}: {error}", flush=True)
                    job.status_text = str(error)
                else:
                    continue
                submit(job, args.commit, args.submission_root)
                active[account].append(job)
                continue
            if state in {"running", "queued", "unknown"}:
                print(f"ADOPT {state.upper()} {job.ref}", flush=True)
                job.status = state
                active[account].append(job)
                continue
            submit(job, args.commit, args.submission_root)
            active[account].append(job)

    for account in ASSIGNMENTS:
        fill(account)
    write_manifest(args.manifest, jobs, args.commit)

    while time.monotonic() < deadline:
        if all(job.status == "complete" for job in jobs):
            break
        for account in ASSIGNMENTS:
            for job in list(active[account]):
                state, status_text = classify_status(job)
                job.status_text = status_text
                if state == "complete":
                    print(f"COMPLETE {job.ref}; downloading", flush=True)
                    try:
                        download(job, args.output_dir)
                    except JobVerificationError as error:
                        print(f"REJECT {job.ref}: {error}", flush=True)
                        job.status_text = str(error)
                        if job.attempts < 2:
                            print(f"RETRY {job.ref}", flush=True)
                            submit(job, args.commit, args.submission_root)
                        else:
                            job.status = "failed"
                            active[account].remove(job)
                    else:
                        active[account].remove(job)
                elif state == "failed":
                    if job.attempts < 2:
                        print(f"RETRY {job.ref}", flush=True)
                        submit(job, args.commit, args.submission_root)
                    else:
                        job.status = "failed"
                        active[account].remove(job)
                else:
                    job.status = state
            fill(account)
        write_manifest(args.manifest, jobs, args.commit)
        completed = sum(job.status == "complete" for job in jobs)
        running = sum(job.status in {"running", "queued", "unknown"} for job in jobs)
        print(f"PROGRESS completed={completed}/25 active={running}", flush=True)
        if any(job.status == "failed" for job in jobs):
            break
        if completed < 25:
            time.sleep(args.poll_seconds)

    write_manifest(args.manifest, jobs, args.commit)
    incomplete = [job.ref for job in jobs if job.status != "complete"]
    if incomplete:
        raise SystemExit(f"V6 {args.method} training incomplete: {incomplete}")
    print(f"V6 25-job {args.method} training audit PASS", flush=True)


if __name__ == "__main__":
    main()
