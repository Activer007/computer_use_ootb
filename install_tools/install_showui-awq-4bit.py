"""Compatibility wrapper for installing the quantised ShowUI weights."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from install_tools.install_showui import run_installation  # noqa: E402


def main() -> None:
    run_installation(precision="awq-4bit", skip_deps=False, force=False, skip_torch=False)


if __name__ == "__main__":
    main()

