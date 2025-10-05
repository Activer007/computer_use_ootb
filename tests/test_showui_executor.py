import pytest
from unittest.mock import patch

from computer_use_demo.executor.showui_executor import ShowUIExecutor
from computer_use_demo.tools import ToolResult


class DummyComputerTool:
    """Minimal stand-in for ComputerTool used during tests."""

    def __init__(self, *_, **__):
        self.calls = []

    def to_params(self):
        return {"name": "computer", "type": "computer_20241022"}

    def sync_call(self, **tool_input):
        self.calls.append(tool_input)
        return ToolResult(output="ok")


@pytest.fixture
def showui_executor():
    with patch.object(ShowUIExecutor, "_get_screen_resolution", return_value=(0, 0, 200, 100)):
        with patch("computer_use_demo.executor.showui_executor.ComputerTool", DummyComputerTool):
            executor = ShowUIExecutor(
                output_callback=lambda *_: None,
                tool_output_callback=lambda *_: None,
                selected_screen=0,
            )
    return executor


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
