"""Enhanced installer for ShowUI weights and runtime dependencies.

The script now handles GPU/MPS detection, recommends the correct PyTorch build,
downloads either the full precision or quantised ShowUI weights, and writes a
small manifest that can be consumed by the Gradio UI.  The goal is to minimise
manual setup steps when preparing ShowUI for local execution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

from huggingface_hub import snapshot_download

# Allow importing project modules when executing the script directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from computer_use_demo.tools.hardware import (  # noqa: E402
    build_performance_plot_data,
    check_showui_assets,
    detect_accelerator,
    dump_environment_report,
    gather_resource_metrics,
    recommend_showui_profile,
    summarise_recommendations,
)


MODEL_REPOS = {
    "fp16": {
        "repo": "showlab/ShowUI-2B",
        "destination": Path("./models/showui-2b"),
    },
    "awq-4bit": {
        "repo": "yyyang/showui-2b-awq",
        "destination": Path("./models/showui-2b-awq"),
    },
}

ENV_REPORT = ROOT / ".showui" / "environment.json"


def _download_model(model_key: str, force: bool = False) -> Path:
    config = MODEL_REPOS[model_key]
    destination = config["destination"].resolve()
    destination.mkdir(parents=True, exist_ok=True)

    print(f"→ Downloading {model_key} weights from {config['repo']} ...")
    snapshot_download(
        repo_id=config["repo"],
        local_dir=destination,
        local_dir_use_symlinks=False,
        resume_download=True,
        force_download=force,
    )
    print(f"✓ Files are available under {destination}")
    return destination


def _run_pip(command: list[str]) -> None:
    from subprocess import CalledProcessError, check_call

    try:
        check_call(command)
    except CalledProcessError as exc:  # pragma: no cover - external process
        raise SystemExit(exc.returncode) from exc


def _install_dependencies(accelerator: Dict[str, str], skip_torch: bool) -> None:
    extras = [
        "transformers>=4.44.0",
        "accelerate>=0.30.0",
        "huggingface_hub>=0.24.0",
        "psutil",
        "pynvml",
    ]

    command = [sys.executable, "-m", "pip", "install", *extras]
    print("→ Ensuring auxiliary packages are installed ...")
    print("  $", " ".join(command))
    _run_pip(command)

    if skip_torch:
        return

    torch_cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
    index_url = accelerator.get("recommended_torch_index_url")
    if index_url:
        torch_cmd.extend(["--index-url", index_url])

    print("→ Installing PyTorch build ...")
    print("  $", " ".join(torch_cmd))
    _run_pip(torch_cmd)


def run_installation(precision: str, skip_deps: bool, force: bool, skip_torch: bool) -> Dict[str, str]:
    accelerator = detect_accelerator()
    recommendation = recommend_showui_profile(accelerator)
    manifest_before = check_showui_assets()

    print("Detected hardware:")
    print(accelerator)
    print()

    if not skip_deps:
        _install_dependencies(accelerator, skip_torch=skip_torch)

    download_path = _download_model(precision, force=force)

    manifest_after = check_showui_assets()
    dump_environment_report(ENV_REPORT, accelerator, recommendation)

    summary = summarise_recommendations(accelerator, recommendation, manifest_after)
    print()
    print(summary)

    print()
    print("Installation summary:")
    print({
        "downloaded": precision,
        "path": str(download_path),
        "previous_manifest": manifest_before,
        "current_manifest": manifest_after,
    })

    metrics = gather_resource_metrics()
    if metrics:
        print("Current resource snapshot:")
        print(metrics)
        print("Plot data:")
        print(build_performance_plot_data(metrics))

    return {"path": str(download_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install ShowUI weights and dependencies")
    parser.add_argument(
        "--precision",
        choices=list(MODEL_REPOS.keys()),
        default="fp16",
        help="Which set of ShowUI weights to download",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Do not install python dependencies",
    )
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="Skip installing torch even if it is missing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_installation(
        precision=args.precision,
        skip_deps=args.skip_deps,
        force=args.force,
        skip_torch=args.skip_torch,
    )


if __name__ == "__main__":
    main()

