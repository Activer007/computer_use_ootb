"""Interactive setup helper for Computer Use OOTB.

This script guides first-time users through the most common setup steps:

* Detect the current Python environment (Conda/venv).
* Install Python dependencies from ``requirements.txt``.
* Optionally download local models (ShowUI full / AWQ) or review UI-TARS & Qwen tips.
* Generate placeholder API key configuration files (``.env`` or ``api_keys.json``).
* Summarise follow-up actions (startup command, ports, etc.).

The goal is to make the onboarding process smoother while still being
transparent about what happens at each step.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from getpass import getpass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_FILE = ROOT / "requirements.txt"


def print_header(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def detect_environment() -> Dict[str, str]:
    """Inspect the current Python runtime and environment details."""

    info = {
        "python_executable": sys.executable,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        info["conda_env"] = os.environ.get("CONDA_DEFAULT_ENV", Path(conda_prefix).name)
    else:
        info["conda_env"] = "<not detected>"

    if hasattr(sys, "real_prefix") or sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        info["virtualenv"] = sys.prefix
    else:
        info["virtualenv"] = "<not detected>"

    info["pip_executable"] = shutil.which("pip") or "<not on PATH>"
    info["conda_executable"] = shutil.which("conda") or "<not on PATH>"

    return info


def show_environment_info() -> None:
    info = detect_environment()
    print_header("Environment check")
    for key, value in info.items():
        print(f"- {key.replace('_', ' ').title()}: {value}")
    print(
        "\nTip: We recommend activating a Conda environment with Python >=3.11 before running this script."
    )


def prompt_yes_no(question: str, default: bool = True) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        answer = input(question + suffix).strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer with 'y' or 'n'.")


def prompt_multi_select(options: Sequence[Tuple[str, str]]) -> List[str]:
    """Allow users to select multiple options by entering comma separated ids."""

    print_header("Optional downloads & helpers")
    print("Enter the numbers of the resources you'd like to prepare (comma-separated).\n"
          "Press Enter to skip.")
    for idx, (_, description) in enumerate(options, start=1):
        print(f"  {idx}. {description}")

    while True:
        raw = input("Selection: ").strip()
        if not raw:
            return []

        indices: List[str] = []
        valid = True
        for item in raw.split(','):
            item = item.strip()
            if not item:
                continue
            if not item.isdigit() or not (1 <= int(item) <= len(options)):
                valid = False
                break
            indices.append(item)

        if valid:
            selected_ids = {options[int(i) - 1][0] for i in indices}
            return sorted(selected_ids)

        print("Invalid selection. Please enter valid option numbers (e.g. 1,3).");


def run_subprocess(command: Sequence[str], description: str) -> bool:
    """Run a subprocess and report success/failure."""

    print(f"\n➡️ {description}\n   Command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"❌ Failed: {exc}")
        return False
    except FileNotFoundError as exc:
        print(f"❌ Missing executable: {exc}")
        return False

    print("✅ Done")
    return True


def install_dependencies() -> bool:
    if not REQUIREMENTS_FILE.exists():
        print(f"requirements.txt not found at {REQUIREMENTS_FILE}. Skipping.")
        return False

    return run_subprocess(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
        "Installing Python dependencies",
    )


def prepare_showui(full_precision: bool) -> bool:
    script = "install_showui.py" if full_precision else "install_showui-awq-4bit.py"
    description = "Downloading ShowUI-2B model" if full_precision else "Downloading ShowUI-2B AWQ 4-bit model"
    script_path = ROOT / "install_tools" / script
    return run_subprocess([sys.executable, str(script_path)], description)


def review_ui_tars() -> None:
    print_header("UI-TARS setup tips")
    print(
        "- UI-TARS requires a separately deployed server (Cloud/VLLM).\n"
        "- Follow the official guide: https://github.com/bytedance/UI-TARS.\n"
        "- Once your server exposes an OpenAI-compatible endpoint, update the OOTB interface\n"
        "  with the base URL and API key. Use install_tools/test_ui-tars_server.py to sanity check\n"
        "  connectivity before launching the UI."
    )


def review_local_qwen() -> None:
    print_header("Local Qwen planner tips")
    print(
        "- You can host a local Qwen planner via computer_use_demo/remote_inference.py.\n"
        "- Recommended hardware: >=24GB RAM, CUDA GPU with >=12GB VRAM for 7B models.\n"
        "- Start the server: python computer_use_demo/remote_inference.py --host 0.0.0.0 --port 8000\n"
        "- Configure OOTB to point to http://<server-ip>:8000/v1 when selecting the planner."
    )


API_KEYS = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "QWEN_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
]


def collect_api_keys() -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not prompt_yes_no("Would you like to enter API keys now?", default=False):
        return {key: "" for key in API_KEYS}

    print("Press Enter to skip any key you don't have yet. Input will be hidden.")
    for key in API_KEYS:
        values[key] = getpass(f"{key}: ")
    return values


def write_env_file(values: Dict[str, str]) -> Path:
    path = ROOT / ".env"
    lines = ["# API keys for Computer Use OOTB"]
    lines.extend(f"{key}={value}" for key, value in values.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_json_file(values: Dict[str, str]) -> Path:
    path = ROOT / "api_keys.json"
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    return path


def configure_api_keys() -> Path | None:
    print_header("API key configuration")
    if not prompt_yes_no("Generate API key template now?", default=True):
        return None

    options = [
        ("env", ".env (dotenv format)"),
        ("json", "api_keys.json (JSON format)"),
    ]
    for idx, (_, label) in enumerate(options, start=1):
        print(f"  {idx}. {label}")

    choice = None
    while choice is None:
        raw = input("Choose file format [1/2]: ").strip() or "1"
        if raw in {"1", "2"}:
            choice = options[int(raw) - 1][0]
        else:
            print("Please enter 1 or 2.")

    values = collect_api_keys()
    if choice == "env":
        path = write_env_file(values)
    else:
        path = write_json_file(values)

    print(f"Saved API key placeholders to {path}")
    return path


def main() -> None:
    print_header("Computer Use OOTB setup")
    show_environment_info()

    performed_steps: List[str] = []
    follow_up: List[str] = []

    if prompt_yes_no("Install dependencies from requirements.txt?", default=True):
        if install_dependencies():
            performed_steps.append("Installed Python dependencies")
        else:
            follow_up.append("Check pip/permission issues if dependencies were not installed.")

    model_options = [
        ("showui_full", "Download ShowUI-2B (full precision)"),
        ("showui_awq", "Download ShowUI-2B AWQ 4-bit (CUDA only)"),
        ("ui_tars", "Review UI-TARS deployment checklist"),
        ("qwen_local", "Review local Qwen planner tips"),
    ]

    selections = prompt_multi_select(model_options)
    for selection in selections:
        if selection == "showui_full":
            if prepare_showui(full_precision=True):
                performed_steps.append("Prepared ShowUI-2B model")
                follow_up.append("Enable ShowUI in the interface and point to the downloaded weights.")
        elif selection == "showui_awq":
            if prepare_showui(full_precision=False):
                performed_steps.append("Prepared ShowUI-2B AWQ model")
                follow_up.append("Select the AWQ quantized option in ShowUI advanced settings (CUDA only).")
        elif selection == "ui_tars":
            review_ui_tars()
            performed_steps.append("Reviewed UI-TARS setup guidance")
        elif selection == "qwen_local":
            review_local_qwen()
            performed_steps.append("Reviewed local Qwen planner guidance")

    api_file = configure_api_keys()
    if api_file:
        performed_steps.append(f"Created API key template at {api_file.relative_to(ROOT)}")
        follow_up.append("Populate the generated file with your actual API keys before launching.")

    follow_up.extend([
        "Launch the interface: python app.py",
        "Default UI port: http://127.0.0.1:7860",
        "Remote (gradio) URL will be printed on launch for secure sharing.",
    ])

    print_header("Summary")
    if performed_steps:
        print("Completed steps:")
        for item in performed_steps:
            print(f"  - {item}")
    else:
        print("No automated steps were executed.")

    print("\nNext actions:")
    for item in follow_up:
        print(f"  • {item}")

    print("\nSetup assistant finished. Happy hacking! ✨")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSetup interrupted by user.")
