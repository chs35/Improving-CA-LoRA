from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from calora.data.dummy import DummyConceptDataset  # noqa: E402
from calora.data.prompts import PromptAugmenter  # noqa: E402
from calora.models.projections import ProjectionTable  # noqa: E402
from calora.models.toy import ToyAdapter  # noqa: E402

AXES = ["style", "viewpoint"]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def augmenter(repo_root: Path) -> PromptAugmenter:
    return PromptAugmenter.from_file(
        repo_root / "configs" / "prompt_augmentations.json", axes=AXES, seed=0
    )


@pytest.fixture
def dataset(augmenter: PromptAugmenter) -> DummyConceptDataset:
    return DummyConceptDataset(augmenter, num_items=4, resolution=64, seed=0)


@pytest.fixture
def adapter() -> ToyAdapter:
    return ToyAdapter(seed=0)


@pytest.fixture
def table(adapter: ToyAdapter) -> ProjectionTable:
    return ProjectionTable.from_model(adapter.unet)
