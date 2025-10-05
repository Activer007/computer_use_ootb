import sys
import types


pyautogui_stub = types.ModuleType("pyautogui")


def _noop(*_args, **_kwargs):
    return None


def _position():
    return 0, 0


pyautogui_stub.scroll = _noop
pyautogui_stub.hscroll = _noop
pyautogui_stub.moveTo = _noop
pyautogui_stub.dragTo = _noop
pyautogui_stub.keyDown = _noop
pyautogui_stub.keyUp = _noop
pyautogui_stub.typewrite = _noop
pyautogui_stub.click = _noop
pyautogui_stub.rightClick = _noop
pyautogui_stub.middleClick = _noop
pyautogui_stub.doubleClick = _noop
pyautogui_stub.mouseDown = _noop
pyautogui_stub.mouseUp = _noop
pyautogui_stub.position = _position

sys.modules.setdefault("pyautogui", pyautogui_stub)
