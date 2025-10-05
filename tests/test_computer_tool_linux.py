import base64
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DISPLAY", ":0")

fake_pyautogui = types.ModuleType("pyautogui")


def _noop(*args, **kwargs):
    return None


fake_pyautogui.moveTo = _noop
fake_pyautogui.dragTo = _noop
fake_pyautogui.position = lambda: (0, 0)
fake_pyautogui.scroll = _noop
fake_pyautogui.hscroll = _noop
fake_pyautogui.click = _noop
fake_pyautogui.rightClick = _noop
fake_pyautogui.middleClick = _noop
fake_pyautogui.doubleClick = _noop
fake_pyautogui.mouseDown = _noop
fake_pyautogui.mouseUp = _noop
fake_pyautogui.typewrite = _noop
fake_pyautogui.keyDown = _noop
fake_pyautogui.keyUp = _noop
fake_pyautogui.FAILSAFE = False

sys.modules.setdefault("pyautogui", fake_pyautogui)

from computer_use_demo.tools import computer
from computer_use_demo.tools.computer import ComputerTool


@pytest.fixture
def linux_tool(monkeypatch, tmp_path):
    captured = {"moves": []}

    monkeypatch.setattr(computer.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        computer.subprocess,
        "check_output",
        lambda *args, **kwargs: b"1920x1080+0+0",
    )
    monkeypatch.setattr(computer, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(computer.time, "sleep", lambda *_, **__: None)

    fake_image = computer.Image.new("RGB", (100, 100), "white")

    def fake_grab(*, all_screens=False, bbox=None):
        captured["bbox"] = bbox
        captured["all_screens"] = all_screens
        return fake_image.copy()

    monkeypatch.setattr(computer.ImageGrab, "grab", fake_grab)

    def fake_move_to(x, y):
        captured["moves"].append((x, y))

    monkeypatch.setattr(computer.pyautogui, "moveTo", fake_move_to)
    monkeypatch.setattr(computer.pyautogui, "position", lambda: (0, 0))
    monkeypatch.setattr(computer.pyautogui, "dragTo", lambda *_, **__: None)
    monkeypatch.setattr(computer.pyautogui, "scroll", lambda *_, **__: None)
    monkeypatch.setattr(computer.pyautogui, "hscroll", lambda *_, **__: None)
    monkeypatch.setattr(computer.pyautogui, "click", lambda *_, **__: None)
    monkeypatch.setattr(computer.pyautogui, "rightClick", lambda *_, **__: None)
    monkeypatch.setattr(computer.pyautogui, "middleClick", lambda *_, **__: None)
    monkeypatch.setattr(computer.pyautogui, "doubleClick", lambda *_, **__: None)
    monkeypatch.setattr(computer.pyautogui, "mouseDown", lambda *_, **__: None)
    monkeypatch.setattr(computer.pyautogui, "mouseUp", lambda *_, **__: None)
    monkeypatch.setattr(computer.pyautogui, "typewrite", lambda *_, **__: None)
    monkeypatch.setattr(computer.pyautogui, "keyDown", lambda *_, **__: None)
    monkeypatch.setattr(computer.pyautogui, "keyUp", lambda *_, **__: None)

    tool = ComputerTool(is_scaling=False)
    tool.target_dimension = computer.MAX_SCALING_TARGETS["WXGA"]
    return tool, captured


@pytest.mark.asyncio
async def test_screenshot_returns_tool_result_with_image(linux_tool):
    tool, captured = linux_tool

    result = await tool.screenshot()

    assert result.base64_image, "Expected a base64 encoded screenshot"
    decoded = base64.b64decode(result.base64_image)
    assert decoded, "Decoded screenshot should not be empty"
    assert captured["bbox"] == (0, 0, 1920, 1080)
    assert captured["all_screens"] is True


@pytest.mark.asyncio
async def test_mouse_move_then_screenshot_still_returns_image(linux_tool):
    tool, captured = linux_tool

    move_result = await tool(
        action="mouse_move",
        coordinate=(100, 200),
    )

    assert "Moved mouse" in move_result.output
    # Should have moved to the absolute coordinates (no offset)
    assert captured["moves"][-1] == (100, 200)

    screenshot_result = await tool.screenshot()

    assert screenshot_result.base64_image, "Expected screenshot after mouse move"
    decoded = base64.b64decode(screenshot_result.base64_image)
    assert decoded, "Decoded screenshot should not be empty"
    assert captured["bbox"] == (0, 0, 1920, 1080)
