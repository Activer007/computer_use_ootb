import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


class DummyComputerTool:
    def __init__(self, *args, **kwargs):
        pass


class DummyToolCollection:
    def __init__(self, *args, **kwargs):
        pass

    def sync_call(self, name, tool_input):  # pragma: no cover - not exercised
        class _Result:
            error = None
            output = ""
            base64_image = None

        return _Result()


@pytest.fixture(autouse=True)
def patch_tool_collection(monkeypatch):
    monkeypatch.setattr(showui_executor_module, "ToolCollection", DummyToolCollection)
    monkeypatch.setattr(showui_executor_module, "ComputerTool", DummyComputerTool)
    monkeypatch.setattr(
        ShowUIExecutor,
        "_get_screen_resolution",
        lambda self: (0, 0, 1920, 1080),
    )


def test_click_coordinates_within_bounds():
    executor = ShowUIExecutor(output_callback=lambda *args, **kwargs: None,
                              tool_output_callback=lambda *args, **kwargs: None)

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
