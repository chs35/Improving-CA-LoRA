"""Where the few-shot campaign is, and when each stage should end, in KST.

    python scripts/ops/campaign_status.py [--run-dir runs/sdxl_cityscapes_fewshot10]

Read-only: parses campaign.log and the stage logs of scripts/ops/run_fewshot10.sh.
A stage's duration comes from, in order: its own progress so far, the mean of the
finished stages of that kind, and the rates measured by the tensor-core gate.

``ARMS`` and the default ``--run-dir`` are those of the earlier few-shot campaign
(2026-11-02), whose CA-selected arms are not kept in this repository. The reported
experiment ran in ``runs/sdxl_cityscapes_selfgen_fewshot10`` with arms A_t81 /
B_top3 / C_t201 x {style, viewpoint}: pass that ``--run-dir``, and note that the
C_t201 arms are not in ``ARMS``, so the table does not list them.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
STAGES = ("train", "generate", "evaluate")
ARMS = (
    "A_t81_style", "A_t81_viewpoint", "B_top3_style", "B_top3_viewpoint", "C_t41_style",
    "C_t41_viewpoint", "D_t1_style", "D_t1_viewpoint", "E_random", "E_all", "control_base",
)
#: Seconds, used until the gate or a finished stage says otherwise.
FALLBACK = {"train": 3.75 * 3600, "generate": 2.25 * 3600, "evaluate": 600.0}
STEP_RE = re.compile(r"step (\d+)/(\d+)\b")
IMAGE_RE = re.compile(r"generate_images \|\s+(\d+)/(\d+)\s*$")
PROBLEM_RE = re.compile(r"Traceback|out of memory|OutOfMemoryError|loss=nan|loss=inf|"
                        r"Input/output error|Errno 5\b")


@dataclass
class StageState:
    status: str = "pending"  # pending | running | done | skipped
    started: datetime | None = None
    ended: datetime | None = None
    attempts: int = 0


@dataclass
class Campaign:
    stages: dict[tuple[str, str], StageState] = field(default_factory=dict)
    halted: str | None = None
    finished: bool = False
    retries: list[str] = field(default_factory=list)


def arm_stages(arm: str) -> tuple[str, ...]:
    return STAGES[1:] if arm == "control_base" else STAGES


def parse_campaign(text: str) -> Campaign:
    campaign = Campaign()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        when = datetime.strptime(parts[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        event = parts[1]
        if event == "HALT":
            campaign.halted = " ".join(parts[2:])
        elif event == "CAMPAIGN" and parts[2] == "END":
            campaign.finished = True
        elif event in ("START", "END", "SKIP", "RETRY"):
            state = campaign.stages.setdefault((parts[2], parts[3]), StageState())
            if event == "START":
                state.status, state.ended = "running", None
                state.attempts += 1
                state.started = state.started or when
            elif event == "END":
                state.ended = when
                if parts[4] == "rc=0":
                    state.status = "done"
            elif event == "SKIP":
                state.status = "skipped"
            else:
                campaign.retries.append(f"{parts[2]} {parts[3]} {parts[4]}")
    return campaign


def progress(stage: str, log_text: str) -> float | None:
    """Fraction of a running stage done, from the last progress line of its log."""
    pattern = STEP_RE if stage == "train" else IMAGE_RE if stage == "generate" else None
    if pattern is None:
        return None
    found = None
    for line in log_text.replace("\r", "\n").splitlines():
        match = pattern.search(line)
        if match:
            found = match
    return int(found.group(1)) / int(found.group(2)) if found else None


def gate_estimates(run_dir: Path) -> dict[str, float]:
    path = run_dir.parent / f"{run_dir.name}_gate" / "tensor_core_gate.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("estimates", {})


def project(campaign: Campaign, run_dir: Path, now: datetime) -> dict[tuple[str, str], datetime]:
    """Expected end of every unfinished stage, walking the campaign's order."""
    defaults = {**FALLBACK, **gate_estimates(run_dir)}
    finished: dict[str, list[float]] = {stage: [] for stage in STAGES}
    for (_, stage), state in campaign.stages.items():
        if state.status == "done" and state.started and state.ended:
            finished[stage].append((state.ended - state.started).total_seconds())

    def typical(stage: str) -> float:
        return sum(finished[stage]) / len(finished[stage]) if finished[stage] else defaults[stage]

    cursor, ends = now, {}
    for arm in ARMS:
        for stage in arm_stages(arm):
            state = campaign.stages.get((arm, stage), StageState())
            if state.status in ("done", "skipped"):
                continue
            remaining = typical(stage)
            if state.status == "running" and state.started:
                elapsed = (now - state.started).total_seconds()
                log = run_dir / "logs" / f"{stage}_{arm}.log"
                text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
                fraction = progress(stage, text)
                if fraction:
                    remaining = elapsed / fraction - elapsed
                else:
                    remaining = max(typical(stage) - elapsed, 0.0)
            cursor += timedelta(seconds=remaining)
            ends[(arm, stage)] = cursor
    return ends


def _shell(command: list[str]) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"(failed: {exc})"


def problems(campaign: Campaign, run_dir: Path) -> list[str]:
    found = []
    if campaign.halted:
        found.append(f"HALT: {campaign.halted}")
    found += [f"retried: {entry}" for entry in campaign.retries]
    for log in sorted((run_dir / "logs").glob("*.log")):
        hits = PROBLEM_RE.findall(log.read_text(encoding="utf-8", errors="replace"))
        if hits:
            found.append(f"{log.name}: {len(hits)} match(es), e.g. {hits[0]}")
    stuck = [
        line for line in _shell(["ps", "-eo", "stat,etime,cmd"]).splitlines()
        if line.startswith("D") and ("scripts/" in line)
    ]
    found += [f"D-state: {line[:120]}" for line in stuck]
    return found


def render(campaign: Campaign, ends: dict, run_dir: Path, now: datetime) -> str:
    kst = lambda moment: moment.astimezone(KST).strftime("%m/%d %H:%M")  # noqa: E731
    lines = [f"as of {kst(now)} KST", "| arm | train | generate | evaluate |", "|---|---|---|---|"]
    for arm in ARMS:
        cells = []
        for stage in STAGES:
            state = campaign.stages.get((arm, stage), StageState())
            if stage not in arm_stages(arm):
                cells.append("-")
            elif state.status in ("done", "skipped"):
                cells.append("done")
            else:
                mark = " (running)" if state.status == "running" else ""
                cells.append(kst(ends[(arm, stage)]) + mark)
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    total = max(ends.values()) if ends else now
    lines.append(f"campaign end: {'finished' if campaign.finished else kst(total)} KST")
    running = [key for key, state in campaign.stages.items() if state.status == "running"]
    for arm, stage in running:
        log = run_dir / "logs" / f"{stage}_{arm}.log"
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        fraction = progress(stage, text)
        lines.append(f"running: {arm} {stage} "
                     + (f"{100 * fraction:.1f}%" if fraction is not None else "(no progress line yet)"))
    workers = [
        line for line in _shell(["ps", "-eo", "stat,etime,cmd"]).splitlines()
        if any(name in line for name in ("train_lora.py", "generate_images.py", "evaluate.py"))
    ]
    lines.append(f"processes: {[line[:60] for line in workers] or 'none'}")
    lines.append("gpu: " + _shell(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                   "--format=csv,noheader"]))
    found = problems(campaign, run_dir)
    lines.append("problems: " + ("; ".join(found) if found else "none"))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/sdxl_cityscapes_fewshot10"))
    args = parser.parse_args()
    log = args.run_dir / "campaign.log"
    text = log.read_text(encoding="utf-8") if log.exists() else ""
    campaign = parse_campaign(text)
    now = datetime.now(timezone.utc)
    print(render(campaign, project(campaign, args.run_dir, now), args.run_dir, now))


if __name__ == "__main__":
    main()
