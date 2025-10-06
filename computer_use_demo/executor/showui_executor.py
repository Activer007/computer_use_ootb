import ast
import asyncio
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
from computer_use_demo.tools.colorful_text import colorful_text_showui, colorful_text_vlm


class ShowUIExecutor:
    def __init__(
        self,
        output_callback: Callable[[BetaContentBlockParam], None],
        tool_output_callback: Callable[[Any, str], None],
        selected_screen: int = 0,
    ):
        self.output_callback = output_callback
        self.tool_output_callback = tool_output_callback
        self.selected_screen = selected_screen
        self.screen_bbox = self._get_screen_resolution()
        print("Screen BBox:", self.screen_bbox)

        self.tool_collection = ToolCollection(
            ComputerTool(selected_screen=selected_screen, is_scaling=False)
        )

        # Supported actions emitted by ShowUI or UI-TARS. We only need
        # membership checks, so keep them in a set.
        self.supported_action_type = {
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
        # Track whether a terminating action (like STOP) was observed so the
        # caller can decide to exit any action loop safely.
        self.stop_requested = False

    def __call__(self, response: str, messages: list[BetaMessageParam]):
        # response is expected to be :
        # {'content': "{'action': 'CLICK', 'value': None, 'position': [0.83, 0.15]}, ...", 'role': 'assistant'},

        action_dict = self._format_actor_output(response)  # str -> dict

        actions = action_dict["content"]
        role = action_dict["role"]

        # Parse the actions from showui
        action_list = self._parse_showui_output(actions)
        print("Parsed Action List:", action_list)

        tool_result_content = None

        if action_list is not None and len(action_list) > 0:

            for action in action_list:  # Execute the tool (adapting the code from anthropic_executor.py)

                tool_result_content: list[BetaToolResultBlockParam] = []

                self.output_callback(f"{colorful_text_showui}:\n{action}", sender="bot")
                print("Converted Action:", action)

                tool_input: dict[str, Any] = dict(action)

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
                # print(f"executor: tool_result_content: {tool_result_content}")
                self.tool_output_callback(result, sim_content_block.id)

                # Craft messages based on the content_block
                # Note: to display the messages in the gradio, you should organize the messages in the following way (user message, bot message)
                display_messages = _message_display_callback(messages)
                # Send the messages to the gradio
                for user_msg, bot_msg in display_messages:
                    yield [user_msg, bot_msg], tool_result_content

        return tool_result_content

    def _format_actor_output(self, action_output: str | dict) -> Dict[str, Any]:
        if isinstance(action_output, dict):
            return action_output
        try:
            action_output.replace("'", '\"')
            action_dict = ast.literal_eval(action_output)
            return action_dict
        except Exception as e:
            print(f"Error parsing action output: {e}")
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

            parsed_output = ast.literal_eval(output_text)

            print("Parsed Output:", parsed_output)

            if isinstance(parsed_output, dict):
                parsed_output = [parsed_output]
            elif not isinstance(parsed_output, list):
                raise ValueError("Parsed output is neither a dictionary nor a list.")

            if not all(isinstance(item, dict) for item in parsed_output):
                raise ValueError("Not all items in the parsed output are dictionaries.")

            # reset termination flag for a new batch of actions
            self.stop_requested = False

            # refine key: value pairs, mapping to the Anthropic's format
            refined_output: list[dict[str, Any]] = []

            def _maybe_resolve_coordinate(action_item: Dict[str, Any]):
                if action_item.get("position") is None:
                    return None
                coordinate = self._resolve_coordinate(action_item)
                action_item["position"] = coordinate
                return coordinate

            for action_item in parsed_output:

                print("Action Item:", action_item)
                # sometime showui returns lower case action names
                action_name = action_item.get("action", "").upper()
                action_item["action"] = action_name

                if action_name not in self.supported_action_type:
                    raise ValueError(
                        f"Action {action_name} not supported. Check the output from ShowUI: {output_text}"
                    )

                if action_name == "STOP":
                    # Signal that execution should stop. We keep any already
                    # generated actions so the caller can finish the current run
                    # but mark that ShowUI requested a stop.
                    self.stop_requested = True
                    break

                if action_name == "CLICK":  # click -> mouse_move + left_click
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

        is_ui_tars = any(tag in {"ui-tars", "ui_tars"} for tag in (position_mode, position_source, source))

        is_absolute = bool(action_item.get("is_absolute")) or position_mode == "absolute"
        if position_source in {"absolute", "ui-tars", "ui_tars"}:
            is_absolute = True
        if source and source in {"ui-tars", "ui_tars"}:
            is_absolute = True

        if not is_absolute and (x_value > 1 or y_value > 1 or x_value < 0 or y_value < 0):
            is_absolute = True

        x_offset = self.screen_bbox[0]
        y_offset = self.screen_bbox[1]
        width = self.screen_bbox[2] - self.screen_bbox[0]
        height = self.screen_bbox[3] - self.screen_bbox[1]

        if is_ui_tars:
            return int(round(x_value)), int(round(y_value))

        if is_absolute:
            return int(round(x_value)) + x_offset, int(round(y_value)) + y_offset

        return (
            int(round(x_value * width)) + x_offset,
            int(round(y_value * height)) + y_offset,
        )

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
                # print(msg["content"][0])
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


# Testing main function
if __name__ == "__main__":
    def output_callback(content_block):
        # print("Output Callback:", content_block)
        pass

    def tool_output_callback(result, action):
        print("[showui_executor] Tool Output Callback:", result, action)
        pass

    # Instantiate the executor
    executor = ShowUIExecutor(
        output_callback=output_callback,
        tool_output_callback=tool_output_callback,
        selected_screen=0,
    )

    # test inputs
    response_content = "{'content': \"{'action': 'CLICK', 'value': None, 'position': [0.49, 0.18]}\", 'role': 'assistant'}"
    # response_content = {'content': "{'action': 'CLICK', 'value': None, 'position': [0.49, 0.39]}", 'role': 'assistant'}
    # response_content = "{'content': \"{'action': 'CLICK', 'value': None, 'position': [0.49, 0.42]}, {'action': 'INPUT', 'value': 'weather for New York city', 'position': [0.49, 0.42]}, {'action': 'ENTER', 'value': None, 'position': None}\", 'role': 'assistant'}"

    # Initialize messages
    messages = []

    # Call the executor
    print("Testing ShowUIExecutor with response content:", response_content)
    for message, tool_result_content in executor(response_content, messages):
        print("Message:", message)
        print("Tool Result Content:", tool_result_content)

    # Display final messages
    print("\nFinal messages:")
    for msg in messages:
        print(msg)

    [
        {
            "role": "user",
            "content": [
                "open a new tab and go to amazon.com",
                "tmp/outputs/screenshot_b4a1b7e60a5c47359bedbd8707573966.png",
            ],
        },
        {"role": "assistant", "content": ["History Action: {'action': 'mouse_move', 'text': None, 'coordinate': (1216, 88)}"]},
        {"role": "assistant", "content": ["History Action: {'action': 'left_click', 'text': None, 'coordinate': None}"]},
        {
            "content": [
                {
                    "type": "tool_result",
                    "content": [
                        {"type": "text", "text": "Moved mouse to (1216, 88)"}
                    ],
                    "tool_use_id": "toolu_ae4f2886-366c-4789-9fa6-ec13461cef12",
                    "is_error": False,
                },
                {
                    "type": "tool_result",
                    "content": [
                        {"type": "text", "text": "Performed left_click"}
                    ],
                    "tool_use_id": "toolu_a7377954-e1b7-4746-9757-b2eb4dcddc82",
                    "is_error": False,
                },
            ],
            "role": "user",
        },
    ]
