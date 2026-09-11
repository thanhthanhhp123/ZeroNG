import pytest

from zerong.remote.vast import (
    REMOTE_EXIT,
    SshTarget,
    offer_query,
    render_job_script,
    select_offer,
    ssh_target,
)


def test_offer_query_uses_vast_gpu_name_syntax():
    q = offer_query("RTX 3090", 0.2, 120)
    assert "gpu_name=RTX_3090" in q
    assert "dph<=0.2" in q and "disk_space>=120" in q and "num_gpus=1" in q


def test_select_offer_cheapest_under_cap():
    offers = [
        {"id": 1, "dph_total": 0.30, "reliability2": 0.99},
        {"id": 2, "dph_total": 0.13, "reliability2": 0.98},
        {"id": 3, "dph_total": 0.13, "reliability2": 0.995},
    ]
    assert select_offer(offers, 0.2)["id"] == 3
    with pytest.raises(RuntimeError):
        select_offer(offers, 0.1)


def test_select_offer_skips_machines_that_failed():
    offers = [
        {"id": 1, "machine_id": 10, "dph_total": 0.13},
        {"id": 2, "machine_id": 10, "dph_total": 0.14},  # same broken host, another offer
        {"id": 3, "machine_id": 20, "dph_total": 0.16},
    ]
    assert select_offer(offers, 0.2, exclude_machines={10})["id"] == 3
    with pytest.raises(RuntimeError):
        select_offer(offers, 0.2, exclude_machines={10, 20})


def test_ssh_target_prefers_direct():
    info = {
        "public_ipaddr": "1.2.3.4 ",
        "ports": {"22/tcp": [{"HostPort": "40022"}]},
        "ssh_host": "ssh5.vast.ai",
        "ssh_port": 12345,
    }
    assert ssh_target(info) == SshTarget("1.2.3.4", 40022)
    assert ssh_target({"ssh_host": "ssh5.vast.ai", "ssh_port": 12345}) == SshTarget(
        "ssh5.vast.ai", 12345
    )
    assert ssh_target({}) is None


def test_job_script_checks_out_exact_commit_and_runs_commands_in_order():
    script = render_job_script(
        "https://github.com/x/ZeroNG.git",
        "abc123",
        ["mvtec_ad", "visa"],
        ["uv run python a.py", "uv run python b.py"],
        max_hours=6,
    )
    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail")
    assert "\r" not in script
    assert f"echo $? > {REMOTE_EXIT}" in script and "trap on_exit EXIT" in script
    assert "git checkout -q abc123" in script
    assert "uv sync --frozen" in script
    assert "download_data.py --dataset mvtec_ad visa" in script
    assert script.index("uv run python a.py") < script.index("uv run python b.py")


def test_job_script_watchdog_stops_but_never_destroys():
    script = render_job_script("https://x/ZeroNG.git", "abc123", [], ["true"], max_hours=6)
    assert f"watchdog {int(6.25 * 3600)} " in script  # hard deadline after launcher timeout
    assert "watchdog 3600 " in script  # grace period after the job ends
    assert '{"state": "stopped"}' in script
    assert "DELETE" not in script and "destroy" not in script
    # The deadline watchdog starts only once curl is installed.
    assert script.index("apt-get install") < script.index(f"watchdog {int(6.25 * 3600)} ")
    # The container key is only read from PID 1's environment, never echoed.
    assert "echo $(env_of CONTAINER_API_KEY)" not in script
