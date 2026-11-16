"""Regenerate the committed synthetic fixture: `uv run python make_synthetic.py`."""

from pathlib import Path

from mimo_golden import synthetic

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "mimo_tiny"
    synthetic.generate(out)
    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"wrote {out} ({total / 1e6:.2f} MB)")
