"""The campaign status report: log parsing and end-time projection."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture(scope="module")
def status(repo_root):
    spec = importlib.util.spec_from_file_location(
        "campaign_status", repo_root / "scripts" / "ops" / "campaign_status.py"
    )
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["campaign_status"] = module
    spec.loader.exec_module(module)
    return module


LOG = """\
2026-11-02T00:00:00Z CAMPAIGN START
2026-11-02T00:00:00Z START A_t81_style train attempt=1
2026-11-02T04:00:00Z END A_t81_style train rc=0
2026-11-02T04:00:00Z START A_t81_style generate attempt=1
"""


def test_the_log_gives_each_stages_state(status):
    campaign = status.parse_campaign(LOG)
    assert campaign.stages[("A_t81_style", "train")].status == "done"
    assert campaign.stages[("A_t81_style", "generate")].status == "running"
    assert not campaign.finished and campaign.halted is None


def test_failures_and_retries_are_surfaced(status):
    text = LOG + (
        "2026-11-02T05:00:00Z END A_t81_style generate rc=1\n"
        "2026-11-02T05:00:00Z RETRY A_t81_style generate rc=1\n"
        "2026-11-02T05:01:00Z HALT A_t81_style generate failed 4 times\n"
    )
    campaign = status.parse_campaign(text)
    assert campaign.retries == ["A_t81_style generate rc=1"]
    assert "failed 4 times" in campaign.halted


def test_progress_reads_the_last_line_of_each_stage(status):
    assert status.progress("train", "step 50/10000 x\nstep 2500/10000 y") == 0.25
    assert status.progress("generate", "generate_images |   450/900\n") == 0.5
    assert status.progress("evaluate", "anything") is None
    assert status.progress("train", "no steps yet") is None


def test_projection_extends_a_running_stage_by_its_own_rate(status, tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "generate_A_t81_style.log").write_text(
        "x | generate_images |   225/900\n", encoding="utf-8"
    )
    campaign = status.parse_campaign(LOG)
    now = datetime(2026, 11, 2, 5, 0, tzinfo=timezone.utc)
    ends = status.project(campaign, tmp_path, now)
    # 1h for a quarter: 3h to go.
    assert ends[("A_t81_style", "generate")] == now + timedelta(hours=3)
    # The next train stage is expected to take as long as the finished one.
    assert ends[("A_t81_viewpoint", "train")] == now + timedelta(hours=3) + timedelta(
        seconds=status.FALLBACK["evaluate"]
    ) + timedelta(hours=4)
    # The control has no train stage and comes last.
    assert ("control_base", "train") not in ends
    assert max(ends.values()) == ends[("control_base", "evaluate")]


def test_render_lists_every_arm(status, tmp_path):
    (tmp_path / "logs").mkdir()
    campaign = status.parse_campaign("")
    now = datetime(2026, 11, 2, 0, 0, tzinfo=timezone.utc)
    text = status.render(campaign, status.project(campaign, tmp_path, now), tmp_path, now)
    for arm in status.ARMS:
        assert arm in text
    assert "problems: none" in text
