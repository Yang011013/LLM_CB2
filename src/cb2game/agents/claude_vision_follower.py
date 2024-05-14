"""This is a template file for creating a new Agent.

Replace the class and method definitions with your own implementations.
"""
import os
os.environ['http_proxy'] = 'http://127.0.0.1:10792'
os.environ['https_proxy'] = 'http://127.0.0.1:10792'

from dataclasses import dataclass
import anthropic
import base64
import concurrent.futures
import functools
import logging
import string
import time

from mashumaro.mixins.json import DataClassJSONMixin

from cb2game.agents.agent import Agent, RateLimitException, Role
from cb2game.pyclient.client_utils import (
    FollowerSystemVisionPrompt,
    SingleActionSystemVisionPrompt,
    crop_non_white_square
)
from cb2game.pyclient.game_endpoint import Action, GameState
from cb2game.server.messages.prop import PropUpdate

logger = logging.getLogger(__name__)

@dataclass
class ClaudeVisionFollowerConfig(DataClassJSONMixin):
    """Configuration for initializing a GeminiFollower agent."""
    api_key: str
    model: str = "claude-3-opus-20240229"
    max_tokens: int = 3000
    temperature: float = 0.4
    queueing_enabled: bool = False


class ClaudeVisionFollower(Agent):
    def __init__(self, config: ClaudeVisionFollowerConfig):
        super().__init__()


        self.temperature = config["temperature"]
        self.model_name = config["model"]
        self.max_tokens = config["max_tokens"]
        self.queueing_enabled = config["queueing_enabled"]

        self.model = anthropic.Anthropic(api_key=config['api_key'])

        self.game_explanation = (# 如果self.queueing_enabled为True，
            FollowerSystemVisionPrompt()
            if self.queueing_enabled
            else SingleActionSystemVisionPrompt()
        )
        self.messages = None

        self.action_queue = []
        self.thought_queue = []
        self.turn_number = 0
        self.action_number = 0
        self.last_instruction = ""

    def role(self) -> Role:
        return Role.FOLLOWER


    def get_image_data(self, image_path):
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        return image_data


    def choose_action(self, game_state: GameState, game=None, action_number=None, action_mask=None) -> Action:

        if len(self.action_queue) > 0 and self.queueing_enabled:
            return self.action_queue.pop(0)
        # Fetch more actions.
        [mapu, props, turn_state, instrs, _, _] = game_state
        if instrs[-1].text != self.last_instruction:
            self.turn_number += 1
            self.last_instruction = instrs[-1].text
        game.visualization().visualize_follower_visibility(self.turn_number, action_number)
        prop_update = PropUpdate(props)
        (leader, follower) = get_actors(game_state)
        self.messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{self.game_explanation},\ninstruction:{instrs[-1].text}"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": self.get_image_data(crop_non_white_square(
                                f"follower_view/follower_visibility_{self.turn_number}_{action_number}.png"))
                        }
                    }
                ]
            },
        ]
        time.sleep(10)
        response = self.model.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            # system=self.game_explanation,
            messages=self.messages
        )

        if not response:
            return Action.NoopAction()

        response_text = response.content[0].text
        print("==============================")
        print(response_text)
        print("==============================")
        action_string = ""
        # Split by lines. If a line starts with "THOUGHTS:" or "THOUGHT:", then
        # print the line. If a line starts with "ACTIONS:" or "ACTION:", then
        # collect everything after the colon and split by comma.
        lines = response_text.split("\n")
        for line in lines:
            if line.startswith("THOUGHTS:") or line.startswith("THOUGHT:"):
                self.thought_queue.append(line.split(":")[1])
            elif line.startswith("ACTIONS:") or line.startswith("ACTION:"):
                action_string = line.split(":")[1]
                # Strip punctuation from the end of the action string.
                action_string = action_string.rstrip(string.punctuation)
            else:
                # Make sure the line isn't just whitespace/punctuation.
                if line.strip(string.whitespace + string.punctuation) == "":
                    continue
                # If the line doesn't start with THOUGHTS or ACTIONS, this is
                # unexpected. GPT is probably not working as expected.
                # logger.warning(f"Unexpected line in Gemini response: {line}.")

        active_instruction = get_active_instruction(instrs)
        actions = actions_from_code(action_string, active_instruction.uuid)
        if len(actions) == 0:
            return Action.NoopAction()

        if self.queueing_enabled:
            self.action_queue = actions[1:]
            return actions[0]
        return actions[0]


# 存储历史信息
def append_messages_to_file(filename, messages, turn_number):
    with open(filename, "a") as file:
        file.write(str(turn_number)+":\n")
        for message in messages:
            file.write(str(message))
        file.write("\n \n")

def timeout_decorator(timeout):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout)
                except concurrent.futures.TimeoutError:
                    print("Function call timed out")
        return wrapper

    return decorator


@timeout_decorator(timeout=120)
def call_gemini_api_sync(messages, model):
    response = model.generate_content(messages)
    return response


def actions_from_code(action_code, i_uuid: str = None):
    # Split action code by comma.
    characters_in_prompt = action_code.split(",")
    if len(characters_in_prompt) == 0:
        # logger.warning("Empty action string.")
        return None
    actions = []
    for c in characters_in_prompt:
        # Convert to lower and strip whitespace.
        c = c.lower().strip()
        if "forward".startswith(c) or "f".startswith(c):
            actions.append(Action.Forwards())
        elif "backward".startswith(c) or "b".startswith(c):
            actions.append(Action.Backwards())
        elif "left".startswith(c) or "l".startswith(c):
            actions.append(Action.Left())
        elif "right".startswith(c) or "r".startswith(c):
            actions.append(Action.Right())
        elif "done".startswith(c) or "d".startswith(c):
            actions.append(Action.InstructionDone(i_uuid))
        else:
            logger.warning(f"Invalid action code: {c}")
    return actions


def get_active_instruction(instructions):
    for instruction in instructions:
        if not instruction.completed and not instruction.cancelled:
            return instruction
    return None


def get_actors(game_state):
    (
        _,
        _,
        _,
        _,
        actors,
        _,
    ) = game_state
    if len(actors) == 1:
        return (None, actors[0])
    else:
        return actors
