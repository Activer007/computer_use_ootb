import json
import types
import sys

import pytest


dummy_pyautogui = types.ModuleType("pyautogui")
dummy_pyautogui.moveTo = lambda *args, **kwargs: None
dummy_pyautogui.position = lambda: (0, 0)
dummy_pyautogui.dragTo = lambda *args, **kwargs: None
dummy_pyautogui.keyDown = lambda *args, **kwargs: None
dummy_pyautogui.keyUp = lambda *args, **kwargs: None
dummy_pyautogui.typewrite = lambda *args, **kwargs: None
dummy_pyautogui.scroll = lambda *args, **kwargs: None
dummy_pyautogui.hscroll = lambda *args, **kwargs: None
dummy_pyautogui.click = lambda *args, **kwargs: None
dummy_pyautogui.rightClick = lambda *args, **kwargs: None
dummy_pyautogui.doubleClick = lambda *args, **kwargs: None
dummy_pyautogui.middleClick = lambda *args, **kwargs: None
dummy_pyautogui.mouseDown = lambda *args, **kwargs: None
dummy_pyautogui.mouseUp = lambda *args, **kwargs: None
dummy_pyautogui.FAILSAFE = False
dummy_pyautogui.PAUSE = 0

sys.modules.setdefault("pyautogui", dummy_pyautogui)
sys.modules.setdefault("mouseinfo", types.ModuleType("mouseinfo"))

from computer_use_demo.executor.showui_executor import ShowUIExecutor
from computer_use_demo.gui_agent.actor.uitars_agent import convert_ui_tars_action_to_json


class MinimalShowUIExecutor(ShowUIExecutor):
    """ShowUIExecutor with a mocked screen resolution and no tool initialization."""

    def __init__(self):
        # Bypass the parent initialisation that depends on local hardware/tools.
        self.screen_bbox = (100, 200, 1100, 1200)
        self.supported_action_type = {
            "CLICK": "key",
            "INPUT": "key",
            "ENTER": "key",
            "ESC": "key",
            "ESCAPE": "key",
            "PRESS": "key",
            "HOVER": "key",
            "SCROLL": "key",
        }

    def output_callback(self, *args, **kwargs):
        pass

    def tool_output_callback(self, *args, **kwargs):
        pass


@pytest.fixture()
def executor():
    return MinimalShowUIExecutor()


def test_parse_showui_normalized_coordinates(executor):
    output = '[{"action": "CLICK", "value": None, "position": [0.5, 0.5]}]'
    actions = executor._parse_showui_output(output)

    assert actions[0]["action"] == "mouse_move"
    assert actions[0]["coordinate"] == (500, 500)
    assert actions[1]["action"] == "left_click"


def test_parse_ui_tars_absolute_coordinates(executor):
    output = str([
        {"action": "CLICK", "value": None, "position": [150, 250], "source": "UI-TARS"}
    ])
    actions = executor._parse_showui_output(output)

    assert actions[0]["coordinate"] == (50, 50)


def test_parse_absolute_coordinates_without_source_flag(executor):
    output = str([
        {"action": "HOVER", "value": None, "position": [1500, 800]}
    ])
    actions = executor._parse_showui_output(output)

    assert actions[0]["action"] == "mouse_move"
    assert actions[0]["coordinate"] == (1400, 600)


def test_convert_ui_tars_action_includes_source_flag():
    json_payload = convert_ui_tars_action_to_json("Action: click(start_box='(153,97)')")
    parsed = json.loads(json_payload)

    assert parsed["source"] == "UI-TARS"
    assert parsed["position"] == [153, 97]
