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


def make_selfgen_set(cfg, directory: Path) -> Path:
    """Generate the toy self-generated CA set ``cfg`` asks for into ``directory``.

    Points ``cfg.measure.selfgen.dir`` at it, so the caller can hand ``cfg`` (or a
    ``--set measure.selfgen.dir=...`` override) to any measurement entry point.
    """
    from calora.data.selfgen import generate_selfgen_set, spec_from_config
    from calora.models.loader import load_adapter

    adapter = load_adapter(cfg.model)
    generate_selfgen_set(
        adapter,
        spec_from_config(cfg, adapter),
        directory,
        num_images=cfg.measure.selfgen.num_images,
        batch_size=cfg.measure.selfgen.batch_size,
    )
    cfg.measure.selfgen.dir = str(directory)
    return directory


@pytest.fixture(scope="session")
def tiny_selfgen_dir(repo_root: Path, tmp_path_factory) -> Path:
    """The tiny_cpu.yaml self-generated set, made once per test session."""
    from calora.config import load_config

    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    return make_selfgen_set(cfg, tmp_path_factory.mktemp("tiny_selfgen"))
