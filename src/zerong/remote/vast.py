"""Run a ZeroNG job on a rented vast.ai GPU and bring the results home.

Design constraints:

- The instance runs code cloned from the (public) git remote at an exact, pushed commit, so every
  result logged remotely traces back to that commit. A dirty or unpushed tree is refused.
- The vast.ai API key never leaves this machine: the instance cannot rent or destroy anything.
- Instances cost money. The launcher destroys the instance when the job ends, fails, times out
  or the launcher is interrupted (unless ``keep``). A state file records the instance id so a
  manual ``destroy`` is possible even if the launcher process itself dies.
- Model checkpoints stay on the instance: only metrics, scores, split manifests, MLflow runs and
  the job log are fetched.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tarfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# vast.ai's own base image (widely used on the platform, so often already on the host). With
# nvidia/cuda:12.6.3-base-ubuntu22.04 from Docker Hub, 3 of the first 5 hosts never finished
# loading the container; the image is a suspected, not proven, cause.
DEFAULT_IMAGE = "vastai/base-image:cuda-12.6.3-auto"
REMOTE_REPO = "/root/ZeroNG"
REMOTE_LOG = "/root/job.log"
REMOTE_EXIT = "/root/job.exit"
PROGRESS_PATTERN = r"run_baseline:\|\[zerong-job\]\|download_data:\|Traceback\|Error"


@dataclass
class JobSpec:
    name: str
    commands: list[str]
    datasets: list[str] = field(default_factory=list)
    gpu_name: str = "RTX 3090"
    max_dph: float = 0.20
    max_hours: float = 8.0
    disk_gb: int = 60
    image: str = DEFAULT_IMAGE
    # Some hosts never start the container (stuck in "created"): give up on a machine after
    # ssh_timeout_min and try the next cheapest one, at most max_attempts machines in total.
    max_attempts: int = 3
    ssh_timeout_min: float = 15.0


@dataclass(frozen=True)
class SshTarget:
    host: str
    port: int
    user: str = "root"


# --------------------------------------------------------------------------- pure helpers


def offer_query(gpu_name: str, max_dph: float, min_disk_gb: int) -> str:
    """vast.ai search query. GPU names use underscores instead of spaces in the query syntax."""
    return (
        f"num_gpus=1 gpu_name={gpu_name.replace(' ', '_')} reliability>0.98 inet_down>300 "
        f"disk_space>={min_disk_gb} cuda_vers>=12.6 verified=true rentable=true dph<={max_dph}"
    )


def select_offer(
    offers: list[dict], max_dph: float, exclude_machines: frozenset | set = frozenset()
) -> dict:
    """Cheapest offer under the price cap, skipping machines that already failed us.

    Ties are broken by higher reliability.
    """
    affordable = [
        o
        for o in offers
        if o.get("dph_total", float("inf")) <= max_dph
        and o.get("machine_id") not in exclude_machines
    ]
    if not affordable:
        raise RuntimeError(f"no offer at or below ${max_dph:.3f}/h")
    return min(affordable, key=lambda o: (o["dph_total"], -o.get("reliability2", 0.0)))


def ssh_target(info: dict) -> SshTarget | None:
    """Prefer a direct connection to the machine; fall back to the vast.ai SSH proxy."""
    direct = (info.get("ports") or {}).get("22/tcp")
    if info.get("public_ipaddr") and direct:
        return SshTarget(info["public_ipaddr"].strip(), int(direct[0]["HostPort"]))
    if info.get("ssh_host") and info.get("ssh_port"):
        return SshTarget(info["ssh_host"], int(info["ssh_port"]))
    return None


# Stops (never destroys) the instance using the key vast.ai scopes to this container, read from
# PID 1's environment. Stopped instances keep their disk, so results stay recoverable.
STOP_SELF_SCRIPT = r"""#!/usr/bin/env bash
env_of() { tr '\0' '\n' < /proc/1/environ | sed -n "s/^$1=//p"; }
curl -s -o /dev/null -w "[zerong-job] self-stop http=%{http_code}\n" -X PUT \
  "https://console.vast.ai/api/v0/instances/$(env_of CONTAINER_ID)/" \
  -H "Authorization: Bearer $(env_of CONTAINER_API_KEY)" \
  -H "Content-Type: application/json" -d '{"state": "stopped"}'
"""
ENV_OF = r"""env_of() { tr '\0' '\n' < /proc/1/environ | sed -n "s/^$1=//p"; }"""
API_CHECK = (
    r"""echo "[zerong-job] watchdog api check http=$(curl -s -o /dev/null -w '%{http_code}' """
    r"""-H "Authorization: Bearer $(env_of CONTAINER_API_KEY)" """
    r""""https://console.vast.ai/api/v0/instances/$(env_of CONTAINER_ID)/?owner=me")" """
)
WATCHDOG_GRACE_HOURS = 1.0


def render_job_script(
    repo_url: str,
    commit: str,
    datasets: list[str],
    commands: list[str],
    max_hours: float,
    grace_hours: float = WATCHDOG_GRACE_HOURS,
) -> str:
    """Bash script run on the instance. Its exit code is written to REMOTE_EXIT.

    Safety net for when the local launcher dies (it was once killed by the OS on low memory and
    the instance sat idle for hours): the instance stops itself at the hard deadline
    (``max_hours`` + 15 min, after the launcher's own timeout), or ``grace_hours`` after the job
    ends if nobody has destroyed it by then.
    """
    deadline_s = int((max_hours + 0.25) * 3600)
    grace_s = int(grace_hours * 3600)
    watchdog = (
        "watchdog() { setsid nohup bash -c \"sleep $1; echo '[zerong-job] $2'; "
        f'bash /root/stop_self.sh" >> {REMOTE_LOG} 2>&1 < /dev/null & }}'
    )
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "export DEBIAN_FRONTEND=noninteractive MLFLOW_DISABLE_AGENT_HINT=1",
        "cat > /root/stop_self.sh <<'STOP'",
        STOP_SELF_SCRIPT.rstrip("\n"),
        "STOP",
        ENV_OF,
        watchdog,
        f"on_exit() {{ echo $? > {REMOTE_EXIT}; watchdog {grace_s} 'job ended, grace over'; }}",
        "trap on_exit EXIT",
        'echo "[zerong-job] setup"',
        "apt-get update -qq",
        "apt-get install -y -qq git curl ca-certificates libglib2.0-0 libgl1 > /dev/null",
        f"watchdog {deadline_s} 'deadline reached'",
        API_CHECK,
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        'export PATH="$HOME/.local/bin:$PATH"',
        f"git clone -q {shlex.quote(repo_url)} {REMOTE_REPO}",
        f"cd {REMOTE_REPO}",
        f"git checkout -q {shlex.quote(commit)}",
        "uv sync --frozen",
        "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
    ]
    if datasets:
        lines.append(
            "uv run python scripts/download_data.py --dataset "
            + " ".join(shlex.quote(d) for d in datasets)
        )
    for cmd in commands:
        lines.append(f"echo {shlex.quote('[zerong-job] $ ' + cmd)}")
        lines.append(cmd)
    lines.append('echo "[zerong-job] done"')
    return "\n".join(lines) + "\n"


def known_bad_machines(runs_dir: Path) -> set:
    """Machines that never became reachable in any previous job under ``runs_dir``."""
    bad = set()
    for state_file in Path(runs_dir).glob("*/job.json"):
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for attempt in state.get("attempts", []):
            if attempt.get("outcome") == "unreachable" and attempt.get("machine_id") is not None:
                bad.add(attempt["machine_id"])
    return bad


# --------------------------------------------------------------------------- side effects


def vastai_bin() -> str:
    exe = os.environ.get("VASTAI_BIN") or shutil.which("vastai")
    if not exe:
        raise RuntimeError("vastai CLI not found: install it or set VASTAI_BIN")
    return exe


def vastai(*args: str, stdin: str | None = None) -> str:
    result = subprocess.run(
        [vastai_bin(), *args], capture_output=True, text=True, input=stdin, timeout=180
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"vastai {' '.join(args[:2])} failed: {detail}")
    return result.stdout


def vastai_json(*args: str):
    out = vastai(*args, "--raw").lstrip("﻿")
    starts = [i for i in (out.find("{"), out.find("[")) if i >= 0]
    if not starts:
        raise RuntimeError(f"vastai {' '.join(args[:2])}: no JSON in output: {out[:200]!r}")
    return json.loads(out[min(starts) :])


def ssh_command(
    target: SshTarget, identity: Path, remote_cmd: str, known_hosts: Path | None = None
) -> list[str]:
    # Instances are short-lived, so their host keys are throwaway. Keep them in a file we own:
    # on Windows, pointing ssh at os.devnull ("nul") creates a literal file named `nul` in the
    # working directory, which git then refuses to index.
    return [
        "ssh",
        "-i", str(identity),
        "-p", str(target.port),
        "-o", "StrictHostKeyChecking=no",
        "-o", f"UserKnownHostsFile={known_hosts or os.devnull}",
        "-o", "ConnectTimeout=20",
        "-o", "ServerAliveInterval=30",
        "-o", "LogLevel=ERROR",
        f"{target.user}@{target.host}",
        remote_cmd,
    ]  # fmt: skip


def ssh(
    target: SshTarget,
    identity: Path,
    remote_cmd: str,
    stdin: str | None = None,
    timeout=300,
    known_hosts: Path | None = None,
) -> subprocess.CompletedProcess:
    # Bytes, not text mode: on Windows, text-mode pipes turn "\n" into "\r\n", and bash then
    # fails on the uploaded job script ("set: pipefail: invalid option name").
    result = subprocess.run(
        ssh_command(target, identity, remote_cmd, known_hosts),
        capture_output=True,
        input=stdin.encode() if stdin is not None else None,
        timeout=timeout,
    )
    return subprocess.CompletedProcess(
        result.args,
        result.returncode,
        result.stdout.decode(errors="replace"),
        result.stderr.decode(errors="replace"),
    )


def pushed_commit(repo_dir: Path) -> tuple[str, str]:
    """(remote URL, HEAD sha), after checking the tree is clean and HEAD is on the remote."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True
        ).stdout.strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("working tree has uncommitted changes: commit and push first")
    sha = git("rev-parse", "HEAD")
    git("fetch", "-q", "origin")
    if not git("branch", "-r", "--contains", sha):
        raise RuntimeError(f"commit {sha[:8]} is not on the remote: push first")
    return git("remote", "get-url", "origin"), sha


class VastJob:
    def __init__(self, spec: JobSpec, state_dir: Path, identity: Path, log: Callable = print):
        self.spec = spec
        self.state_dir = Path(state_dir)
        self.identity = Path(identity)
        self.log = log
        self.state_file = self.state_dir / "job.json"
        self.known_hosts = self.state_dir / "known_hosts"
        self.state: dict = {}

    def save(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=1), encoding="utf-8")

    def run(self, repo_dir: Path, keep: bool = False) -> int | None:
        if self.state_file.exists():
            previous = json.loads(self.state_file.read_text(encoding="utf-8"))
            if previous.get("instance_id") and not previous.get("destroyed"):
                raise RuntimeError(
                    f"instance {previous['instance_id']} from a previous run is not destroyed"
                )
        repo_url, commit = pushed_commit(repo_dir)
        self.state = {
            "spec": asdict(self.spec),
            "commit": commit,
            "repo": repo_url,
            "attempts": [],
            "status": "provisioning",
        }
        failed_machines = known_bad_machines(self.state_dir.parent)
        if failed_machines:
            self.log(f"skipping machines that failed before: {sorted(failed_machines)}")
        for _ in range(self.spec.max_attempts):
            offer = self.create_instance(failed_machines)
            started = time.monotonic()
            try:
                target = self.wait_for_ssh(timeout_s=self.spec.ssh_timeout_min * 60)
                break
            except TimeoutError:
                machine = offer.get("machine_id")
                self.log(f"machine {machine} unreachable: destroying it, trying another")
                self.close_attempt(offer, started, "unreachable", keep=False)
                failed_machines.add(offer.get("machine_id"))
            except BaseException as exc:
                self.state["status"] = f"error: {exc!r}"[:300]
                self.close_attempt(offer, started, "error", keep)
                raise
        else:
            self.state["status"] = "error: no reachable machine"
            self.save()
            raise RuntimeError(f"no reachable machine after {self.spec.max_attempts} attempts")

        exit_code: int | None = None
        job_started = False
        try:
            script = render_job_script(
                repo_url, commit, self.spec.datasets, self.spec.commands, self.spec.max_hours
            )
            self.check(
                ssh(
                    target,
                    self.identity,
                    "cat > /root/job.sh",
                    stdin=script,
                    known_hosts=self.known_hosts,
                )
            )
            self.check(
                ssh(
                    target,
                    self.identity,
                    f"nohup setsid bash /root/job.sh > {REMOTE_LOG} 2>&1 < /dev/null &",
                    known_hosts=self.known_hosts,
                )
            )
            job_started = True
            self.state["status"] = "running"
            self.save()
            deadline = started + self.spec.max_hours * 3600
            exit_code = self.poll(target, deadline)
            self.fetch(target)
            self.state["status"] = (
                "timeout" if exit_code is None else "succeeded" if exit_code == 0 else "failed"
            )
            self.state["exit_code"] = exit_code
        except BaseException as exc:
            self.state["status"] = f"error: {exc!r}"[:300]
            if job_started:
                # Keep whatever the job produced before the instance is destroyed.
                try:
                    self.fetch(target)
                except Exception as fetch_exc:
                    self.log(f"could not fetch partial results: {fetch_exc!r}")
            raise
        finally:
            self.close_attempt(offer, started, self.state["status"], keep)
            self.state["finished_at"] = datetime.now(UTC).isoformat()
            self.save()
            self.log(
                f"status={self.state['status']} attempts={len(self.state['attempts'])} "
                f"hours={self.state['hours']:.2f} est_cost=${self.state['est_cost_usd']:.2f} "
                f"destroyed={self.state['destroyed']}"
            )
        return exit_code

    def create_instance(self, exclude_machines: set) -> dict:
        query = offer_query(self.spec.gpu_name, self.spec.max_dph, self.spec.disk_gb)
        # --storage makes dph_total include the disk we will rent (the default prices 5 GB only).
        offers = vastai_json(
            "search", "offers", query, "-o", "dph", "--storage", str(self.spec.disk_gb)
        )
        offer = select_offer(offers, self.spec.max_dph, exclude_machines)
        self.log(
            f"offer {offer['id']} (machine {offer.get('machine_id')}): {offer['gpu_name']} "
            f"${offer['dph_total']:.3f}/h {offer.get('geolocation', '')}"
        )
        created = vastai_json(
            "create", "instance", str(offer["id"]),
            "--image", self.spec.image,
            "--disk", str(self.spec.disk_gb),
            "--ssh", "--direct",
            "--label", f"zerong-{self.spec.name}",
        )  # fmt: skip
        self.state.update(
            offer={
                k: offer.get(k)
                for k in ("id", "machine_id", "gpu_name", "dph_total", "geolocation")
            },
            instance_id=created["new_contract"],
            started_at=datetime.now(UTC).isoformat(),
            status="starting",
            destroyed=False,
        )
        self.save()
        instance_id, commit = self.state["instance_id"], self.state["commit"]
        self.log(f"instance {instance_id} created for commit {commit[:8]}")
        return offer

    def close_attempt(self, offer: dict, started: float, outcome: str, keep: bool) -> None:
        """Record one rented machine's time and estimated cost, then destroy it (unless keep)."""
        hours = (time.monotonic() - started) / 3600
        self.state["attempts"].append(
            {
                "instance_id": self.state.get("instance_id"),
                "machine_id": offer.get("machine_id"),
                "outcome": outcome,
                "hours": round(hours, 3),
                "est_cost_usd": round(hours * offer["dph_total"], 3),
            }
        )
        if not keep:
            self.destroy()
        attempts = self.state["attempts"]
        self.state["hours"] = round(sum(a["hours"] for a in attempts), 3)
        self.state["est_cost_usd"] = round(sum(a["est_cost_usd"] for a in attempts), 3)
        self.save()

    @staticmethod
    def check(result: subprocess.CompletedProcess) -> None:
        if result.returncode != 0:
            raise RuntimeError(f"remote command failed: {result.stderr.strip()[:300]}")

    def wait_for_ssh(self, timeout_s: float) -> SshTarget:
        deadline = time.monotonic() + timeout_s
        last_status = None
        while time.monotonic() < deadline:
            info = vastai_json("show", "instance", str(self.state["instance_id"]))
            status = info.get("actual_status")
            if status != last_status:
                self.log(f"instance status: {status}")
                last_status = status
            target = ssh_target(info) if status == "running" else None
            if target:
                probe = ssh(target, self.identity, "true", timeout=40, known_hosts=self.known_hosts)
                if probe.returncode == 0:
                    self.log(f"ssh ready: {target.user}@{target.host}:{target.port}")
                    return target
            time.sleep(15)
        raise TimeoutError("instance did not become reachable over ssh in time")

    def poll(self, target: SshTarget, deadline: float, every_s: int = 60) -> int | None:
        seen: set[str] = set()
        failures = 0
        # Only the tail of the log is scanned, so each probe stays cheap as the log grows.
        probe = (
            f"cat {REMOTE_EXIT} 2>/dev/null; echo '---'; "
            f"tail -c 2000000 {REMOTE_LOG} | grep -a '{PROGRESS_PATTERN}' | tail -n 20"
        )
        while time.monotonic() < deadline:
            try:
                result = ssh(target, self.identity, probe, timeout=60, known_hosts=self.known_hosts)
                ok = result.returncode == 0
            except (subprocess.TimeoutExpired, OSError):
                # A slow or dropped connection is not a job failure. (An uncaught timeout here
                # once made the launcher destroy an instance in the middle of a sweep.)
                ok = False
            if not ok:
                failures += 1
                self.log(f"progress probe failed ({failures}/10)")
                if failures > 10:
                    raise RuntimeError("lost ssh connection to the instance")
            else:
                failures = 0
                head, _, progress = result.stdout.partition("---")
                for line in progress.strip().splitlines():
                    if line not in seen:
                        seen.add(line)
                        self.log(f"remote | {line}")
                if head.strip():
                    return int(head.strip())
            time.sleep(every_s)
        self.log("max_hours reached: stopping")
        return None

    def fetch(self, target: SshTarget) -> None:
        archive = self.state_dir / "results.tgz"
        remote = (
            f"tar czf - --ignore-failed-read --exclude='*/anomalib' -C {REMOTE_REPO} "
            f"results mlflow.db mlartifacts -C /root job.log 2>/dev/null"
        )
        with archive.open("wb") as f:
            subprocess.run(
                ssh_command(target, self.identity, remote, self.known_hosts),
                stdout=f,
                timeout=3600,
                check=False,
            )
        with tarfile.open(archive) as tar:
            tar.extractall(self.state_dir, filter="data")
        self.log(f"results fetched to {self.state_dir}")

    def destroy(self) -> None:
        instance_id = self.state.get("instance_id")
        if instance_id and not self.state.get("destroyed"):
            vastai("destroy", "instance", str(instance_id), stdin="y\n")
            self.state["destroyed"] = True
            self.log(f"instance {instance_id} destroyed")
