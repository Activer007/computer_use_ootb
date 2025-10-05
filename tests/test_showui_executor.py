import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

computer_module = types.ModuleType("computer_use_demo.tools.computer")


class _DummyComputerTool:  # pragma: no cover - simple stub for imports
    def __init__(self, *args, **kwargs):
        pass


computer_module.ComputerTool = _DummyComputerTool
sys.modules.setdefault("computer_use_demo.tools.computer", computer_module)

from computer_use_demo.executor.showui_executor import ShowUIExecutor


def _make_executor():
    executor = ShowUIExecutor.__new__(ShowUIExecutor)
    executor.screen_bbox = (0, 0, 100, 100)
    executor.supported_action_type = {
        "CLICK",
        "INPUT",
        "ENTER",
        "ESC",
        "ESCAPE",
        "PRESS",
        "HOVER",
        "SCROLL",
        "HOTKEY",
        "STOP",
    }
    executor.stop_requested = False
    return executor


def test_hover_scroll_and_hotkey_are_parsed():
    executor = _make_executor()

    actions = executor._parse_showui_output(
        str([
            {"action": "hover", "position": [0.5, 0.25]},
            {
                "action": "scroll",
                "value": {"direction": "down", "amount": 20},
                "position": [0.1, 0.2],
            },
            {"action": "hotkey", "value": ["CTRL", "L"]},
        ])
    )

    assert actions == [
        {"action": "mouse_move", "text": None, "coordinate": (50, 25)},
        {
            "action": "scroll",
            "text": None,
            "coordinate": (10, 20),
            "scroll_direction": "down",
            "scroll_amount": 20,
        },
        {"action": "key", "text": "ctrl+l", "coordinate": None},
    ]
    assert executor.stop_requested is False


def test_stop_action_sets_flag_and_returns_empty():
    executor = _make_executor()

    actions = executor._parse_showui_output(str([{"action": "STOP"}]))

    assert actions == []
    assert executor.stop_requested is True

    # Follow-up parse should reset the flag
    follow_up_actions = executor._parse_showui_output(str([{"action": "hover", "position": [0.0, 0.0]}]))
    assert executor.stop_requested is False
    assert follow_up_actions == [{"action": "mouse_move", "text": None, "coordinate": (0, 0)}]


def test_hotkey_string_normalization():
    executor = _make_executor()

    actions = executor._parse_showui_output(str([{"action": "hotkey", "value": "CTRL+SHIFT+P"}]))

    assert actions == [{"action": "key", "text": "ctrl+shift+p", "coordinate": None}]
