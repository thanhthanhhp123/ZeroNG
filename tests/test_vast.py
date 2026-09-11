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
    )
    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail")
    assert f"> {REMOTE_EXIT}' EXIT" in script  # exit code always recorded
    assert "git checkout -q abc123" in script
    assert "uv sync --frozen" in script
    assert "download_data.py --dataset mvtec_ad visa" in script
    assert script.index("uv run python a.py") < script.index("uv run python b.py")
