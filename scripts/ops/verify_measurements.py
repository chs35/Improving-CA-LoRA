"""Re-read every measurement of a run and prove it is intact (the t=41 mfs lesson).

    python scripts/ops/verify_measurements.py --run-dir runs/ca_selfgen \
        --timesteps 1 41 61 ... --expect-source selfgen

Each ``t*.safetensors`` (full and halves) is loaded back through
:func:`calora.ca.stats.load_measurement` at head granularity, every tensor is
checked finite, the expected timesteps (and both halves of each) must all be
present, and every file must carry the same provenance. A sha256 list of the
files is written next to them (``measurements/SHA256SUMS``) so a copy made for
another project can be checked with ``sha256sum -c``. Exit status 1 on any
failure.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from calora.ca.stats import common_provenance, load_measurement  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(run_dir: Path, timesteps: list[int] | None, granularity: str,
          expect_source: str | None, halves: bool) -> list[str]:
    directory = run_dir / "measurements"
    errors: list[str] = []
    paths = sorted(directory.glob("t*.safetensors"))
    if not paths:
        return [f"no measurements in {directory}"]
    loaded = {}
    for path in paths:
        try:
            result = load_measurement(path, expect_granularity=granularity)
        except Exception as exc:  # noqa: BLE001 - every failure is reported, not raised
            errors.append(f"{path.name}: unreadable ({exc})")
            continue
        tensors = [result.gd_norm_mean, *result.cos_of_means.values(),
                   *result.mean_of_cos.values()]
        for stats in result.per_axis.values():
            tensors += [stats.ca_mean, stats.ca_std, stats.gc_norm_mean]
        if not all(bool(torch.isfinite(t).all()) for t in tensors):
            errors.append(f"{path.name}: non-finite values")
        if f"t{result.timestep:04d}" != path.stem.split("_")[0]:
            errors.append(f"{path.name}: header says t={result.timestep}")
        loaded[path.name] = result
    if timesteps:
        for t in timesteps:
            names = [f"t{t:04d}.safetensors"]
            if halves:
                names += [f"t{t:04d}_half0.safetensors", f"t{t:04d}_half1.safetensors"]
            errors += [f"missing {name}" for name in names if name not in loaded]
    try:
        provenance = common_provenance(list(loaded.values()))
        if expect_source and provenance.get("data_source") != expect_source:
            errors.append(f"data_source={provenance.get('data_source')!r}, "
                          f"expected {expect_source!r}")
    except ValueError as exc:
        errors.append(str(exc))
    if not errors:
        lines = [f"{sha256(p)}  {p.name}" for p in paths]
        (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--timesteps", type=int, nargs="*", default=None)
    parser.add_argument("--granularity", default="head")
    parser.add_argument("--expect-source", default=None)
    parser.add_argument("--no-halves", action="store_true")
    args = parser.parse_args()
    errors = check(args.run_dir, args.timesteps, args.granularity, args.expect_source,
                   halves=not args.no_halves)
    for error in errors:
        print(f"FAIL {error}")
    if errors:
        return 1
    print(f"OK {args.run_dir / 'measurements'} (SHA256SUMS written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
