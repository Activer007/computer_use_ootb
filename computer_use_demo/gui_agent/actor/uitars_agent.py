import json
import re
from openai import OpenAI

from computer_use_demo.gui_agent.llm_utils.oai import encode_image
from computer_use_demo.tools.screen_capture import get_screenshot
from computer_use_demo.tools.logger import logger, truncate_string


class UITARS_Actor:
    """
    In OOTB, we use the default grounding system prompt form UI_TARS repo, and then convert its action to our action format.
    """

    _NAV_SYSTEM_GROUNDING = """
You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```Action: ...```

## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use \"\" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.

## Note
- Do not generate any other text.
"""

    def __init__(self, ui_tars_url, output_callback, api_key="", selected_screen=0):

        self.ui_tars_url = ui_tars_url
        self.ui_tars_client = OpenAI(base_url=self.ui_tars_url, api_key=api_key)
        self.selected_screen = selected_screen
        self.output_callback = output_callback

        self.grounding_system_prompt = self._NAV_SYSTEM_GROUNDING.format()


    def __call__(self, messages):

        task = messages
        
        # take screenshot
        screenshot, screenshot_path = get_screenshot(selected_screen=self.selected_screen, resize=True, target_width=1920, target_height=1080)
        screenshot_width, screenshot_height = screenshot.size
        screenshot_path = str(screenshot_path)
        screenshot_base64 = encode_image(screenshot_path)

        logger.info(f"Sending messages to UI-TARS on {self.ui_tars_url}: {task}, screenshot: {screenshot_path}")

        response = self.ui_tars_client.chat.completions.create(
            model="ui-tars",
            messages=[
                {"role": "system", "content": self.grounding_system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": task},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}}
                    ]
                },
                ],
            max_tokens=256,
            temperature=0
            )
        
        ui_tars_action = response.choices[0].message.content
        converted_action = convert_ui_tars_action_to_json(
            ui_tars_action,
            screenshot_size=(screenshot_width, screenshot_height),
        )
        response = str(converted_action)

        response = {'content': response, 'role': 'assistant'}
        return response



def convert_ui_tars_action_to_json(action_str: str, screenshot_size: tuple[int, int] | None = None) -> str:
    """Convert a UI-TARS action line into the ShowUI JSON schema."""

    action_str = action_str.strip()
    if action_str.startswith("Action:"):
        action_str = action_str[len("Action:"):].strip()

    action_map = {
        "click": "CLICK",
        "hover": "HOVER",
        "press": "PRESS",
        "type": "INPUT",
        "scroll": "SCROLL",
        "wait": "STOP",
        "finished": "STOP",
        "call_user": "STOP",
        "hotkey": "HOTKEY",
    }

    payload: dict[str, object] = {
        "action": None,
        "value": None,
        "position": None,
        "position_source": "ui-tars",
    }

    def _normalize_position(raw_x: int, raw_y: int) -> list[float] | list[int]:
        if not screenshot_size:
            return [int(raw_x), int(raw_y)]

        width, height = screenshot_size
        if width <= 0 or height <= 0:
            return [int(raw_x), int(raw_y)]

        clamped_x = max(0, min(int(raw_x), width - 1))
        clamped_y = max(0, min(int(raw_y), height - 1))

        return [clamped_x / width, clamped_y / height]

    position_match = re.match(r"^(click|hover|press)\(start_box='\(?(\d+),\s*(\d+)\)?'\)$", action_str)
    if position_match:
        action_name, x, y = position_match.groups()
        payload["action"] = action_map[action_name]
        payload["position"] = _normalize_position(int(x), int(y))
        if not screenshot_size:
            payload["position_mode"] = "absolute"
        return json.dumps(payload)

    hotkey_match = re.match(r"^hotkey\(key='([^']+)'\)$", action_str)
    if hotkey_match:
        key = hotkey_match.group(1).lower()
        if key == "enter":
            payload["action"] = "ENTER"
        elif key == "esc":
            payload["action"] = "ESC"
        else:
            payload["action"] = action_map["hotkey"]
            payload["value"] = key
        return json.dumps(payload)

    type_match = re.match(r"^type\(content='([^']*)'\)$", action_str)
    if type_match:
        payload["action"] = action_map["type"]
        payload["value"] = type_match.group(1)
        return json.dumps(payload)

    scroll_match = re.match(
        r"^scroll\(start_box='[^']*'\s*,\s*direction='(down|up|left|right)'\)$",
        action_str,
    )
    if scroll_match:
        payload["action"] = action_map["scroll"]
        payload["value"] = scroll_match.group(1)
        return json.dumps(payload)

    if action_str in {"wait()", "finished()", "call_user()"}:
        base_action = action_str.replace("()", "")
        payload["action"] = action_map.get(base_action, "STOP")
        return json.dumps(payload)

    payload["action"] = "STOP"
    return json.dumps(payload)