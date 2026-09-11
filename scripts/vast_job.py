"""Run a ZeroNG job on a rented vast.ai GPU (see zerong.remote.vast for the guarantees).

Requires the `vastai` CLI (on PATH or via VASTAI_BIN) with an API key configured, and an SSH key
registered on the vast.ai account. HEAD must be committed and pushed.

Examples:
    uv run python scripts/vast_job.py run --name patchcore-baseline --max-dph 0.2 --max-hours 8 \
        --datasets mvtec_ad visa \
        --cmd "uv run python scripts/run_baseline.py --config configs/patchcore_mvtec.yaml"
    uv run python scripts/vast_job.py status --name patchcore-baseline
    uv run python scripts/vast_job.py destroy --name patchcore-baseline
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from zerong.remote.vast import DEFAULT_IMAGE, JobSpec, VastJob, vastai_json

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "remote_runs"


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)

    run = sub.add_parser("run", help="rent, run, fetch results, destroy")
    run.add_argument("--name", required=True)
    run.add_argument("--cmd", action="append", required=True, help="command run in the repo")
    run.add_argument("--datasets", nargs="*", default=[])
    run.add_argument("--gpu", default="RTX 3090")
    run.add_argument("--max-dph", type=float, default=0.20, help="price cap, $/hour")
    run.add_argument("--max-hours", type=float, default=8.0)
    run.add_argument("--disk", type=int, default=60, help="GB (priced into the offer search)")
    run.add_argument("--image", default=DEFAULT_IMAGE)
    run.add_argument("--identity", type=Path, default=Path.home() / ".ssh" / "id_ed25519_vast")
    run.add_argument("--keep", action="store_true", help="do not destroy the instance at the end")

    for name in ("status", "destroy"):
        p = sub.add_parser(name)
        p.add_argument("--name", required=True)

    args = parser.parse_args()
    state_dir = RUNS / args.name

    if args.action == "run":
        spec = JobSpec(
            name=args.name,
            commands=args.cmd,
            datasets=args.datasets,
            gpu_name=args.gpu,
            max_dph=args.max_dph,
            max_hours=args.max_hours,
            disk_gb=args.disk,
            image=args.image,
        )
        exit_code = VastJob(spec, state_dir, args.identity, log).run(REPO, keep=args.keep)
        raise SystemExit(0 if exit_code == 0 else 1)

    state = json.loads((state_dir / "job.json").read_text(encoding="utf-8"))
    if args.action == "status":
        print(json.dumps(state, indent=1))
        if not state.get("destroyed"):
            info = vastai_json("show", "instance", str(state["instance_id"]))
            print("live:", {k: info.get(k) for k in ("actual_status", "dph_total", "gpu_name")})
    else:
        job = VastJob(JobSpec(name=args.name, commands=[]), state_dir, Path(), log)
        job.state = state
        job.destroy()
        job.save()


if __name__ == "__main__":
    main()
