"""The run-artefact plots: a loss curve is written, and nothing else can fail a run."""

from __future__ import annotations

import pytest

from calora.train.lora import _format_duration
from calora.utils.plotting import plot_loss_curve


def _history(num_entries: int = 10, *, decay: bool = False) -> list[dict]:
    return [
        {
            "step": 10 * (i + 1),
            "loss": 1.0 / (i + 1),
            "learning_rate": 1e-4 * (1 - i / num_entries) if decay else 1e-4,
            "elapsed": 2.5 * (i + 1),
        }
        for i in range(num_entries)
    ]


# -- loss curve ------------------------------------------------------------------------------


def test_plot_loss_curve_writes_a_png(tmp_path):
    pytest.importorskip("matplotlib")

    path = plot_loss_curve(_history(), tmp_path / "loss_curve.png")

    assert path is not None and path.exists()
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_loss_curve_plots_a_varying_learning_rate_too(tmp_path):
    """The lr panel only appears when the schedule moves; both paths have to draw."""
    pytest.importorskip("matplotlib")

    path = plot_loss_curve(_history(decay=True), tmp_path / "decayed.png")

    assert path is not None and path.stat().st_size > 0


def test_plot_loss_curve_creates_missing_parent_directories(tmp_path):
    pytest.importorskip("matplotlib")

    path = plot_loss_curve(_history(), tmp_path / "nested" / "loss_curve.png")

    assert path is not None and path.exists()


def test_an_empty_history_produces_no_file(tmp_path):
    """`log_every=0` means no history at all; that is not an error."""
    target = tmp_path / "loss_curve.png"
    assert plot_loss_curve([], target) is None
    assert not target.exists()


# -- ETA formatting ---------------------------------------------------------------------------


def test_format_duration_is_h_mm_ss():
    assert _format_duration(0) == "0:00:00"
    assert _format_duration(59.4) == "0:00:59"
    assert _format_duration(3600 + 2 * 60 + 3) == "1:02:03"
    assert _format_duration(360000) == "100:00:00"


def test_format_duration_clamps_a_negative_remaining_time():
    assert _format_duration(-5.0) == "0:00:00"
