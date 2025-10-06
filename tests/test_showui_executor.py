import json
from unittest.mock import patch

import pytest

from computer_use_demo.executor.showui_executor import ShowUIExecutor
from computer_use_demo.gui_agent.actor.uitars_agent import convert_ui_tars_action_to_json
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
        {"action": "key", "text": "ctrl+c", "coordinate": None}
    ]


def test_hotkey_string_normalization(showui_executor):
    parsed = showui_executor._parse_showui_output(
        '[{"action": "hotkey", "value": "CTRL+SHIFT+P"}]'
    )

    assert parsed == [
        {"action": "key", "text": "ctrl+shift+p", "coordinate": None}
    ]


def test_hover_scroll_and_hotkey_are_parsed(showui_executor):
    actions = showui_executor._parse_showui_output(
        str(
            [
                {"action": "hover", "position": [0.5, 0.25]},
                {
                    "action": "scroll",
                    "value": {"direction": "down", "amount": 20},
                    "position": [0.1, 0.2],
                },
                {"action": "hotkey", "value": ["CTRL", "L"]},
            ]
        )
    )

    assert actions == [
        {"action": "mouse_move", "text": None, "coordinate": (100, 25)},
        {
            "action": "scroll",
            "text": None,
            "coordinate": (20, 20),
            "scroll_direction": "down",
            "scroll_amount": 20,
        },
        {"action": "key", "text": "ctrl+l", "coordinate": None},
    ]
    assert showui_executor.stop_requested is False


def test_stop_terminates_parsed_actions(showui_executor):
    parsed = showui_executor._parse_showui_output(
        '[{"action": "click", "position": [0.4, 0.6], "value": None},'
        ' {"action": "stop"},'
        ' {"action": "input", "value": "ignored"}]'
    )

    assert parsed == [
        {"action": "mouse_move", "text": None, "coordinate": (80, 60)},
        {"action": "left_click", "text": None, "coordinate": None},
    ]
    assert showui_executor.stop_requested is True


def test_stop_flag_resets_after_follow_up_parse(showui_executor):
    stop_actions = showui_executor._parse_showui_output(str([{"action": "STOP"}]))
    assert stop_actions == []
    assert showui_executor.stop_requested is True

    follow_up_actions = showui_executor._parse_showui_output(
        str([{"action": "hover", "position": [0.0, 0.0]}])
    )
    assert showui_executor.stop_requested is False
    assert follow_up_actions == [
        {"action": "mouse_move", "text": None, "coordinate": (0, 0)}
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
            "scroll_amount": 10,
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


def test_absolute_coordinates_shifted_by_screen_offset():
    with patch.object(ShowUIExecutor, "_get_screen_resolution", return_value=(100, 50, 500, 450)):
        with patch("computer_use_demo.executor.showui_executor.ComputerTool", DummyComputerTool):
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


def test_ui_tars_normalized_positions_are_scaled(showui_executor):
    parsed = showui_executor._parse_showui_output(
        '[{"action": "click", "position": [0.5, 0.5], "position_mode": "normalized", "position_source": "ui-tars"}]'
    )

    assert parsed == [
        {"action": "mouse_move", "text": None, "coordinate": (100, 50)},
        {"action": "left_click", "text": None, "coordinate": None},
    ]


def test_phone_swipe_maps_to_drag():
    metadata = {
        "selected_screen": 0,
        "raw_size": (200, 400),
        "processed_size": (1080, 1920),
        "crop_box": (20, 0, 180, 400),
    }
    with patch.object(ShowUIExecutor, "_get_screen_resolution", return_value=(0, 0, 200, 400)):
        with patch("computer_use_demo.executor.showui_executor.ComputerTool", DummyComputerTool):
            executor = ShowUIExecutor(
                output_callback=lambda *_: None,
                tool_output_callback=lambda *_: None,
                selected_screen=0,
                split="phone",
            )

    with patch("computer_use_demo.executor.showui_executor.get_last_screenshot_info", return_value=metadata):
        parsed = executor._parse_showui_output(
            '[{"action": "swipe", "position": [[0.1, 0.2], [0.9, 0.8]]}]'
        )

    assert parsed == [
        {"action": "mouse_move", "text": None, "coordinate": (36, 80)},
        {"action": "left_click_drag", "text": None, "coordinate": (164, 320)},
    ]


def test_convert_ui_tars_action_includes_source_flag_and_normalizes():
    json_payload = convert_ui_tars_action_to_json(
        "Action: click(start_box='(960,540)')",
        screenshot_size=(1920, 1080),
    )
    parsed = json.loads(json_payload)

    assert parsed["source"] == "UI-TARS"
    assert parsed["position_source"] == "ui-tars"
    assert parsed["action"] == "CLICK"
    assert parsed["position_mode"] == "normalized"
    assert parsed["position"] == [0.5, 0.5]


def test_convert_ui_tars_action_absolute_when_size_missing():
    json_payload = convert_ui_tars_action_to_json("Action: click(start_box='(153,97)')")
    parsed = json.loads(json_payload)

    assert parsed["position"] == [153, 97]
    assert parsed["position_mode"] == "absolute"
    assert parsed["source"] == "UI-TARS"
