import json
import sys
import types
from unittest.mock import patch

import pytest

# Provide lightweight stand-ins for optional GUI dependencies used by the executor.
dummy_pyautogui = types.ModuleType("pyautogui")
dummy_pyautogui.FAILSAFE = False
dummy_pyautogui.size = lambda: (1920, 1080)
dummy_pyautogui.position = lambda: (0, 0)
dummy_pyautogui.moveTo = lambda *args, **kwargs: None
dummy_pyautogui.click = lambda *args, **kwargs: None
dummy_pyautogui.dragTo = lambda *args, **kwargs: None
dummy_pyautogui.mouseDown = lambda *args, **kwargs: None
dummy_pyautogui.mouseUp = lambda *args, **kwargs: None
dummy_pyautogui.typewrite = lambda *args, **kwargs: None
dummy_pyautogui.press = lambda *args, **kwargs: None
sys.modules.setdefault("pyautogui", dummy_pyautogui)
sys.modules.setdefault("mouseinfo", types.ModuleType("mouseinfo"))

from computer_use_demo.executor import showui_executor as showui_executor_module
from computer_use_demo.executor.showui_executor import ShowUIExecutor
from computer_use_demo.gui_agent.actor.uitars_agent import convert_ui_tars_action_to_json


class _DummyToolResult:
    def __init__(self, output: str = "", error: str | None = None, base64_image: str | None = None):
        self.output = output
        self.error = error
        self.base64_image = base64_image
        self.system = None


class DummyComputerTool:
    """Minimal stand-in for ComputerTool used during tests."""

    def __init__(self, *args, **kwargs):
        self.calls: list[dict[str, object]] = []

    def sync_call(self, **tool_input):  # pragma: no cover - not exercised
        self.calls.append(tool_input)
        return _DummyToolResult(output="ok")

    def to_params(self):  # pragma: no cover - not exercised
        return {"name": "computer", "type": "computer_20241022"}


class DummyToolCollection:
    def __init__(self, *args, **kwargs):
        pass

    def sync_call(self, name, tool_input):  # pragma: no cover - not exercised
        return _DummyToolResult(output="ok")


@pytest.fixture(autouse=True)
def patch_tool_collection(monkeypatch):
    monkeypatch.setattr(showui_executor_module, "ToolCollection", DummyToolCollection)
    monkeypatch.setattr(showui_executor_module, "ComputerTool", DummyComputerTool)
    monkeypatch.setattr(
        ShowUIExecutor,
        "_get_screen_resolution",
        lambda self: (0, 0, 1920, 1080),
    )


@pytest.fixture
def showui_executor(monkeypatch):
    monkeypatch.setattr(ShowUIExecutor, "_get_screen_resolution", lambda self: (0, 0, 200, 100))
    return ShowUIExecutor(
        output_callback=lambda *_: None,
        tool_output_callback=lambda *_: None,
        selected_screen=0,
    )


def test_supported_actions_include_stop_and_hotkey(showui_executor):
    assert "STOP" in showui_executor.supported_action_type
    assert "HOTKEY" in showui_executor.supported_action_type
    assert showui_executor.supported_action_type["HOTKEY"] == "key"


def test_parse_hotkey_into_key_action(showui_executor):
    parsed = showui_executor._parse_showui_output(
        '[{"action": "hotkey", "value": "Ctrl+C"}]'
    )
    assert parsed == [
        {"action": "key", "text": "Ctrl+C", "coordinate": None}
    ]


def test_stop_terminates_parsed_actions(showui_executor):
    parsed = showui_executor._parse_showui_output(
        '[{"action": "click", "position": [0.4, 0.6], "value": None},'
        ' {"action": "stop"},'
        ' {"action": "input", "value": "ignored"}]'
    )
    # click should map to move + click, but input after stop should be ignored
    assert parsed == [
        {"action": "mouse_move", "text": None, "coordinate": (80, 60)},
        {"action": "left_click", "text": None, "coordinate": None},
    ]


def test_scroll_includes_horizontal_directions(showui_executor):
    parsed = showui_executor._parse_showui_output(
        '[{"action": "scroll", "value": "left", "position": [0.25, 0.75]}]'
    )
    assert parsed == [
        {
            "action": "scroll",
            "text": None,
            "coordinate": (50, 75),
            "scroll_direction": "left",
        }
    ]


def test_click_with_absolute_coordinates_skips_scaling(showui_executor):
    parsed = showui_executor._parse_showui_output(
        '[{"action": "click", "position": [150, 40]}]'
    )

    assert parsed == [
        {"action": "mouse_move", "text": None, "coordinate": (150, 40)},
        {"action": "left_click", "text": None, "coordinate": None},
    ]


def test_hover_respects_position_markers_from_ui_tars(showui_executor):
    parsed = showui_executor._parse_showui_output(
        '[{"action": "hover", "position": [321, 222], "position_mode": "absolute", "position_source": "ui-tars"}]'
    )

    assert parsed == [
        {"action": "mouse_move", "text": None, "coordinate": (321, 222)}
    ]


def test_absolute_coordinates_shifted_by_screen_offset(monkeypatch):
    monkeypatch.setattr(ShowUIExecutor, "_get_screen_resolution", lambda self: (100, 50, 500, 450))
    executor = ShowUIExecutor(
        output_callback=lambda *_: None,
        tool_output_callback=lambda *_: None,
        selected_screen=0,
    )

    parsed = executor._parse_showui_output(
        '[{"action": "click", "position": [150, 40]}]'
    )

    assert parsed == [
        {"action": "mouse_move", "text": None, "coordinate": (250, 90)},
        {"action": "left_click", "text": None, "coordinate": None},
    ]


def test_click_coordinates_within_bounds():
    executor = ShowUIExecutor(output_callback=lambda *_: None, tool_output_callback=lambda *_: None)

    action_json = convert_ui_tars_action_to_json(
        "Action: click(start_box='(960,540)')",
        screenshot_size=(1920, 1080),
    )

    payload = json.loads(action_json)
    assert payload["position"] == [0.5, 0.5]

    refined_actions = executor._parse_showui_output(action_json)
    mouse_moves = [action for action in refined_actions if action["action"] == "mouse_move"]
    assert len(mouse_moves) == 1

    x, y = mouse_moves[0]["coordinate"]
    assert 0 <= x < 1920
    assert 0 <= y < 1080


def test_parse_showui_fallback_preserves_string_literals():
    executor = ShowUIExecutor(output_callback=lambda *_: None, tool_output_callback=lambda *_: None)

    # Single quotes require the sanitized literal-eval fallback
    action_json = "[{'action': 'input', 'value': \"literal true\", 'position': [0.1, 0.2]}]"

    refined_actions = executor._parse_showui_output(action_json)
    assert refined_actions == [
        {"action": "type", "text": "literal true", "coordinate": None}
    ]


def test_parse_showui_fallback_converts_json_literals_outside_strings():
    executor = ShowUIExecutor(output_callback=lambda *_: None, tool_output_callback=lambda *_: None)

    action_json = "[{\"action\": \"click\", \"value\": null, \"position\": [0.25, 0.75]}]"

    refined_actions = executor._parse_showui_output(action_json)

    assert refined_actions[0]["action"] == "mouse_move"
    assert refined_actions[0]["coordinate"] == (480, 810)
    assert refined_actions[1] == {"action": "left_click", "text": None, "coordinate": None}
