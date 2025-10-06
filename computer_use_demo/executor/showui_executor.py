import ast
import asyncio
import json
import uuid
from collections.abc import Callable
from typing import Any, Dict, List, Union, cast

from anthropic.types import TextBlock
from anthropic.types.beta import (
    BetaContentBlock,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlock,
)

from computer_use_demo.tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult
from computer_use_demo.tools.screen_capture import get_last_screenshot_info
from computer_use_demo.tools.colorful_text import colorful_text_showui, colorful_text_vlm


class ShowUIExecutor:
    def __init__(
        self,
        output_callback: Callable[[BetaContentBlockParam], None],
        tool_output_callback: Callable[[Any, str], None],
        selected_screen: int = 0,
        split: str = "desktop",
    ):
        self.output_callback = output_callback
        self.tool_output_callback = tool_output_callback
        self.selected_screen = selected_screen
        self.split = split
        self.screen_bbox = self._get_screen_resolution()
        print("Screen BBox:", self.screen_bbox)

        self.tool_collection = ToolCollection(
            ComputerTool(selected_screen=selected_screen, is_scaling=False)
        )

        # Supported actions emitted by ShowUI or UI-TARS.
        self.supported_action_type = {
            "CLICK": "mouse",
            "HOVER": "mouse",
            "INPUT": "type",
            "ENTER": "key",
            "ESC": "key",
            "ESCAPE": "key",
            "PRESS": "mouse",
            "SCROLL": "scroll",
            "HOTKEY": "key",
            "STOP": None,
            "TAP": "mouse",
            "SWIPE": "mouse",
            "ANSWER": "type",
        }
        # Track whether a terminating action (like STOP) was observed so the
        # caller can decide to exit any action loop safely.
        self.stop_requested = False

    def __call__(self, response: str, messages: list[BetaMessageParam]):
        # response is expected to be :
        # {'content': "{'action': 'CLICK', 'value': None, 'position': [0.83, 0.15]}, ...", 'role': 'assistant'},

        action_dict = self._format_actor_output(response)
        if not action_dict:
            return None

        actions = action_dict.get("content")
        if actions is None:
            return None

        # Parse the actions from showui
        action_list = self._parse_showui_output(actions)
        print("Parsed Action List:", action_list)

        tool_result_content = None

        if action_list is not None and len(action_list) > 0:

            for action in action_list:  # Execute the tool (adapting the code from anthropic_executor.py)

                tool_result_content = []

                self.output_callback(f"{colorful_text_showui}:\n{action}", sender="bot")
                print("Converted Action:", action)

                tool_input: Dict[str, Any] = {
                    "action": action["action"],
                    "text": action.get("text"),
                    "coordinate": action.get("coordinate"),
                }
                if "scroll_direction" in action:
                    tool_input["scroll_direction"] = action["scroll_direction"]
                if "scroll_amount" in action:
                    tool_input["scroll_amount"] = action["scroll_amount"]

                sim_content_block = BetaToolUseBlock(
                    id=f"toolu_{uuid.uuid4()}",
                    input=tool_input,
                    name="computer",
                    type="tool_use",
                )

                # update messages
                new_message = {
                    "role": "assistant",
                    "content": cast(list[BetaContentBlockParam], [sim_content_block]),
                }
                if new_message not in messages:
                    messages.append(new_message)

                # Run the asynchronous tool execution in a synchronous context
                result = self.tool_collection.sync_call(
                    name=sim_content_block.name,
                    tool_input=cast(dict[str, Any], sim_content_block.input),
                )

                tool_result_content.append(
                    _make_api_tool_result(result, sim_content_block.id)
                )
                self.tool_output_callback(result, sim_content_block.id)

                # Craft messages based on the content_block
                display_messages = _message_display_callback(messages)
                for user_msg, bot_msg in display_messages:
                    yield [user_msg, bot_msg], tool_result_content

        return tool_result_content

    def _format_actor_output(self, action_output: str | dict) -> Dict[str, Any] | None:
        if isinstance(action_output, dict):
            return action_output

        if not isinstance(action_output, str):
            print(f"Unexpected action output type: {type(action_output)}")
            return None

        text_output = action_output.strip()

        if not text_output:
            print("Empty action output received from ShowUI actor.")
            return None

        try:
            return json.loads(text_output)
        except json.JSONDecodeError:
            pass

        try:
            sanitized_output = self._json_literals_to_python(text_output)
            return ast.literal_eval(sanitized_output)
        except (ValueError, SyntaxError) as exc:
            print(f"Error parsing action output: {exc}")
            return None

    def _parse_showui_output(self, output_text: str) -> Union[List[Dict[str, Any]], None]:
        try:
            output_text = output_text.strip()

            # process single dictionary
            if output_text.startswith("{") and output_text.endswith("}"):
                output_text = f"[{output_text}]"

            # Validate if the output resembles a list of dictionaries
            if not (output_text.startswith("[") and output_text.endswith("]")):
                raise ValueError("Output does not look like a valid list or dictionary.")

            print("Output Text:", output_text)

            try:
                parsed_output = json.loads(output_text)
            except json.JSONDecodeError:
                sanitized_output = self._json_literals_to_python(output_text)
                parsed_output = ast.literal_eval(sanitized_output)

            print("Parsed Output:", parsed_output)

            if isinstance(parsed_output, dict):
                parsed_output = [parsed_output]
            elif not isinstance(parsed_output, list):
                raise ValueError("Parsed output is neither a dictionary nor a list.")

            if not all(isinstance(item, dict) for item in parsed_output):
                raise ValueError("Not all items in the parsed output are dictionaries.")

            # reset termination flag for a new batch of actions
            self.stop_requested = False

            refined_output: list[Dict[str, Any]] = []
            stop_encountered = False

            def _maybe_resolve_coordinate(action_item: Dict[str, Any]):
                if action_item.get("position") is None:
                    return None
                coordinate = self._resolve_coordinate(action_item)
                action_item["position"] = coordinate
                return coordinate

            for action_item in parsed_output:

                print("Action Item:", action_item)
                # sometime showui returns lower case action names
                action_name = (action_item.get("action") or "").upper()
                action_item["action"] = action_name

                if action_name not in self.supported_action_type:
                    raise ValueError(
                        f"Action {action_name} not supported. Check the output from ShowUI: {output_text}"
                    )

                if action_name == "STOP":
                    stop_encountered = True
                    break

                if action_name == "CLICK" or action_name == "TAP":  # click/tap -> mouse_move + left_click
                    coordinate = _maybe_resolve_coordinate(action_item)
                    if coordinate is not None:
                        refined_output.append({"action": "mouse_move", "text": None, "coordinate": coordinate})
                    refined_output.append({"action": "left_click", "text": None, "coordinate": None})

                elif action_name == "INPUT":  # input -> type
                    refined_output.append({"action": "type", "text": action_item.get("value"), "coordinate": None})

                elif action_name == "ENTER":  # enter -> key, enter
                    refined_output.append({"action": "key", "text": "Enter", "coordinate": None})

                elif action_name in {"ESC", "ESCAPE"}:  # escape -> key, escape
                    refined_output.append({"action": "key", "text": "Escape", "coordinate": None})

                elif action_name == "HOVER":  # hover -> mouse_move
                    coordinate = _maybe_resolve_coordinate(action_item)
                    if coordinate is not None:
                        refined_output.append({"action": "mouse_move", "text": None, "coordinate": coordinate})

                elif action_name == "SCROLL":  # scroll -> ComputerTool scroll
                    scroll_value = action_item.get("value")
                    scroll_direction = None
                    scroll_amount = 10
                    scroll_coordinate = _maybe_resolve_coordinate(action_item)

                    if isinstance(scroll_value, dict):
                        scroll_direction = scroll_value.get("direction")
                        if "amount" in scroll_value and scroll_value["amount"] is not None:
                            scroll_amount = int(scroll_value["amount"])
                    elif isinstance(scroll_value, (list, tuple)) and scroll_value:
                        scroll_direction = scroll_value[0]
                        if len(scroll_value) > 1 and isinstance(scroll_value[1], (int, float)):
                            scroll_amount = int(scroll_value[1])
                    elif isinstance(scroll_value, str):
                        scroll_direction = scroll_value

                    if not scroll_direction:
                        raise ValueError("Scroll direction missing or invalid.")

                    scroll_direction = str(scroll_direction).lower()
                    if scroll_direction not in {"up", "down", "left", "right"}:
                        raise ValueError(f"Scroll direction {scroll_direction} not supported.")

                    refined_output.append(
                        {
                            "action": "scroll",
                            "text": None,
                            "coordinate": scroll_coordinate,
                            "scroll_direction": scroll_direction,
                            "scroll_amount": int(scroll_amount),
                        }
                    )

                elif action_name == "PRESS":  # press -> mouse_move + left_press
                    coordinate = _maybe_resolve_coordinate(action_item)
                    if coordinate is not None:
                        refined_output.append({"action": "mouse_move", "text": None, "coordinate": coordinate})
                    refined_output.append({"action": "left_press", "text": None, "coordinate": None})

                elif action_name == "SWIPE":
                    swipe_path = action_item.get("position")
                    if not isinstance(swipe_path, (list, tuple)) or len(swipe_path) != 2:
                        raise ValueError("SWIPE action requires start and end positions.")
                    start_coord = self._resolve_coordinate({**action_item, "position": swipe_path[0]})
                    end_coord = self._resolve_coordinate({**action_item, "position": swipe_path[1]})
                    refined_output.append({"action": "mouse_move", "text": None, "coordinate": start_coord})
                    refined_output.append({"action": "left_click_drag", "text": None, "coordinate": end_coord})

                elif action_name == "ANSWER":
                    answer_text = action_item.get("value") or action_item.get("text")
                    if not isinstance(answer_text, str):
                        raise ValueError("ANSWER action requires textual value.")
                    refined_output.append({"action": "type", "text": answer_text, "coordinate": None})

                elif action_name == "HOTKEY":
                    hotkey_value = action_item.get("value")
                    keys: list[str]
                    if isinstance(hotkey_value, str):
                        keys = [part.strip() for part in hotkey_value.split("+") if part.strip()]
                    elif isinstance(hotkey_value, (list, tuple)):
                        keys = [str(key).strip() for key in hotkey_value if str(key).strip()]
                    else:
                        raise ValueError("Hotkey value must be a string or list of keys.")

                    if not keys:
                        raise ValueError("Hotkey value is empty.")

                    hotkey_text = "+".join(key.lower() for key in keys)

                    refined_output.append({"action": "key", "text": hotkey_text, "coordinate": None})

            if stop_encountered:
                self.stop_requested = True
                return refined_output

            return refined_output

        except Exception as e:
            print(f"Error parsing output: {e}")
            return None

    def _resolve_coordinate(self, action_item: Dict[str, Any]) -> tuple[int, int]:
        position = action_item.get("position")
        if position is None:
            raise ValueError(f"Action {action_item['action']} requires a position but none was provided.")

        if not isinstance(position, (list, tuple)) or len(position) != 2:
            raise ValueError(f"Invalid position payload: {position}")

        x_raw, y_raw = position

        try:
            x_value = float(x_raw)
            y_value = float(y_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Position values must be numeric: {position}") from exc

        position_mode = (action_item.get("position_mode") or "").lower()
        position_source = (action_item.get("position_source") or "").lower()
        source = (action_item.get("source") or "").lower()

        is_absolute = bool(action_item.get("is_absolute")) or position_mode == "absolute"
        if position_source in {"absolute"}:
            is_absolute = True
        if source in {"absolute"}:
            is_absolute = True
        if position_mode in {"normalized", "relative"}:
            is_absolute = False

        if not is_absolute and (x_value > 1 or y_value > 1 or x_value < 0 or y_value < 0):
            is_absolute = True

        viewport_bbox, image_size = self._get_viewport_bbox()
        x_offset, y_offset, x_max, y_max = viewport_bbox
        width = x_max - x_offset
        height = y_max - y_offset
        image_width, image_height = image_size

        if width <= 0 or height <= 0:
            raise ValueError("Invalid screen bounds returned from monitor lookup.")

        if is_absolute:
            x_px = int(round(x_value))
            y_px = int(round(y_value))
            if image_width and image_width > 0 and image_height and image_height > 0:
                scale_x = width / image_width
                scale_y = height / image_height
                x_px = int(round(x_px * scale_x))
                y_px = int(round(y_px * scale_y))
            return x_px + x_offset, y_px + y_offset

        x_clamped = max(0.0, min(1.0, x_value))
        y_clamped = max(0.0, min(1.0, y_value))

        if image_width and image_width > 0 and image_height and image_height > 0:
            scale_x = width / image_width
            scale_y = height / image_height
            x_px = int(round(x_clamped * image_width * scale_x))
            y_px = int(round(y_clamped * image_height * scale_y))
        else:
            x_px = int(round(x_clamped * width))
            y_px = int(round(y_clamped * height))

        x_px = min(max(x_px, 0), width - 1)
        y_px = min(max(y_px, 0), height - 1)

        return x_px + x_offset, y_px + y_offset

    def _get_viewport_bbox(self) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
        """Determine the active viewport used for action mapping."""
        info = get_last_screenshot_info() or {}
        bbox = self.screen_bbox

        if self.split == "phone" and info.get("selected_screen") == self.selected_screen:
            crop_box = info.get("crop_box")
            if crop_box:
                bbox = (
                    bbox[0] + crop_box[0],
                    bbox[1] + crop_box[1],
                    bbox[0] + crop_box[2],
                    bbox[1] + crop_box[3],
                )

        image_size = info.get("processed_size") or info.get("resized_size") or info.get("raw_size")
        if not image_size:
            image_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])

        return bbox, image_size

    def _get_screen_resolution(self):
        from screeninfo import get_monitors
        import platform
        if platform.system() == "Darwin":
            import Quartz  # uncomment this line if you are on macOS
        import subprocess

        # Detect platform
        system = platform.system()

        if system == "Windows":
            # Windows: Use screeninfo to get monitor details
            screens = get_monitors()

            # Sort screens by x position to arrange from left to right
            sorted_screens = sorted(screens, key=lambda s: s.x)

            if self.selected_screen < 0 or self.selected_screen >= len(screens):
                raise IndexError("Invalid screen index.")

            screen = sorted_screens[self.selected_screen]
            bbox = (screen.x, screen.y, screen.x + screen.width, screen.y + screen.height)

        elif system == "Darwin":  # macOS
            # macOS: Use Quartz to get monitor details
            max_displays = 32  # Maximum number of displays to handle
            active_displays = Quartz.CGGetActiveDisplayList(max_displays, None, None)[1]

            # Get the display bounds (resolution) for each active display
            screens = []
            for display_id in active_displays:
                bounds = Quartz.CGDisplayBounds(display_id)
                screens.append(
                    {
                        "id": display_id,
                        "x": int(bounds.origin.x),
                        "y": int(bounds.origin.y),
                        "width": int(bounds.size.width),
                        "height": int(bounds.size.height),
                        "is_primary": Quartz.CGDisplayIsMain(display_id),  # Check if this is the primary display
                    }
                )

            # Sort screens by x position to arrange from left to right
            sorted_screens = sorted(screens, key=lambda s: s["x"])

            if self.selected_screen < 0 or self.selected_screen >= len(screens):
                raise IndexError("Invalid screen index.")

            screen = sorted_screens[self.selected_screen]
            bbox = (screen["x"], screen["y"], screen["x"] + screen["width"], screen["y"] + screen["height"])

        else:  # Linux or other OS
            cmd = "xrandr | grep ' primary' | awk '{print $4}'"
            try:
                output = subprocess.check_output(cmd, shell=True).decode()
                resolution = output.strip()
                # Parse the resolution format like "1920x1080+1920+0"
                # The format is "WIDTHxHEIGHT+X+Y"
                parts = resolution.split("+")[0]  # Get just the "1920x1080" part
                width, height = map(int, parts.split("x"))
                # Get the X, Y offset if needed
                x_offset = int(resolution.split("+")[1]) if len(resolution.split("+")) > 1 else 0
                y_offset = int(resolution.split("+")[2]) if len(resolution.split("+")) > 2 else 0
                bbox = (x_offset, y_offset, x_offset + width, y_offset + height)
            except subprocess.CalledProcessError:
                raise RuntimeError("Failed to get screen resolution on Linux.")

        return bbox

    def _json_literals_to_python(self, text: str) -> str:
        """Convert JSON literal tokens to Python equivalents without touching quoted strings."""

        def is_escaped(idx: int) -> bool:
            backslash_count = 0
            j = idx - 1
            while j >= 0 and text[j] == "\\":
                backslash_count += 1
                j -= 1
            return backslash_count % 2 == 1

        result: list[str] = []
        in_single_quote = False
        in_double_quote = False
        i = 0
        length = len(text)

        while i < length:
            ch = text[i]

            if ch == "'" and not in_double_quote and not is_escaped(i):
                in_single_quote = not in_single_quote
                result.append(ch)
                i += 1
                continue

            if ch == '"' and not in_single_quote and not is_escaped(i):
                in_double_quote = not in_double_quote
                result.append(ch)
                i += 1
                continue

            if not in_single_quote and not in_double_quote:
                if text.startswith("null", i):
                    result.append("None")
                    i += 4
                    continue
                if text.startswith("true", i):
                    result.append("True")
                    i += 4
                    continue
                if text.startswith("false", i):
                    result.append("False")
                    i += 5
                    continue

            result.append(ch)
            i += 1

        return ''.join(result)



def _message_display_callback(messages):
    display_messages = []
    for msg in messages:
        try:
            if isinstance(msg["content"][0], TextBlock):
                display_messages.append((msg["content"][0].text, None))  # User message
            elif isinstance(msg["content"][0], BetaTextBlock):
                display_messages.append((None, msg["content"][0].text))  # Bot message
            elif isinstance(msg["content"][0], BetaToolUseBlock):
                display_messages.append(
                    (None, f"Tool Use: {msg['content'][0].name}\nInput: {msg['content'][0].input}")
                )  # Bot message
            elif isinstance(msg["content"][0], Dict) and msg["content"][0]["content"][-1]["type"] == "image":
                display_messages.append(
                    (None, f'<img src="data:image/png;base64,{msg["content"][0]["content"][-1]["source"]["data"]}">')
                )  # Bot message
            else:
                pass
        except Exception as e:
            print("error", e)
            pass
    return display_messages


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
