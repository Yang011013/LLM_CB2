"""This is a template file for creating a new Agent.

Replace the class and method definitions with your own implementations.
"""
import os
os.environ['http_proxy'] = 'http://127.0.0.1:10792'
os.environ['https_proxy'] = 'http://127.0.0.1:10792'
from dataclasses import dataclass
from pathlib import Path
import google.generativeai as genai
import re
import concurrent.futures
import functools
import logging
import string

import tiktoken
from mashumaro.mixins.json import DataClassJSONMixin

from cb2game.agents.agent import Agent, RateLimitException, Role
from cb2game.pyclient.client_utils import (
    DescribeMap,
    FollowerSystemVisionPrompt,
    SingleActionSystemVisionPrompt,
    crop_non_white_square
)
from cb2game.pyclient.game_endpoint import Action, GameState
from cb2game.server.messages.prop import PropUpdate

logger = logging.getLogger(__name__)

@dataclass
class GeminiVisionFollowerConfig(DataClassJSONMixin):
    """Configuration for initializing a GeminiFollower agent."""
    gemini_api_key: str
    model: str = "gemini-1.0-vision-pro"
    max_output_tokens: int = (
        4096  # Model maximum of 12288. Completion consumes some tokens too though.
    )
    max_in_tokens: int = 30000
    temperature: float = 0.4
    top_p: int = 1
    top_k: int = 32
    queueing_enabled: bool = False


class GeminiVisionFollower(Agent):
    def __init__(self, config: GeminiVisionFollowerConfig):
        super().__init__()


        self.temperature = config["temperature"]
        self.top_p = config["top_p"]
        self.top_k = config["top_k"]
        genai.configure(api_key=config["gemini_api_key"])
        self.model_name = config["model"]
        self.max_output_tokens = config["max_output_tokens"]
        self.max_in_tokens = config["max_in_tokens"]
        self.queueing_enabled = config["queueing_enabled"]
        self.safety_settings = [
              {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
              },
              {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
              },
              {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
              },
              {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
              },
        ]

        self.gemini_vision_config = {
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "top_k": config["top_k"],
            "max_output_tokens": config["max_output_tokens"],
        }
        # self.model = genai.GenerativeModel(model_name=self.model_name,
        #                                    generation_config=self.gemini_vision_config,
        #                                    safety_settings=self.safety_settings)
        self.model = genai.GenerativeModel(model_name=self.model_name,
                                           generation_config=self.gemini_vision_config)
        self.game_explanation = FollowerSystemVisionPrompt() if self.queueing_enabled else SingleActionSystemVisionPrompt()

        self.image_parts = []
        self.instruction_parts = []
        self.messages = None

        self.action_queue = []
        self.turn_number = 0
        self.action_number = 0
        self.last_instruction = ""

    def role(self) -> Role:
        return Role.FOLLOWER


    def _count_tokens(self, text: str):
        tokens = self.model.count_tokens(text)
        return tokens.total_tokens


    def choose_action(self, game_state: GameState, game=None, action_number=None, action_mask=None) -> Action:

        if len(self.action_queue) > 0 and self.queueing_enabled:
            return self.action_queue.pop(0)
        # Fetch more actions.
        [mapu, props, turn_state, instrs, _, _] = game_state

        prop_update = PropUpdate(props)
        (leader, follower) = get_actors(game_state)
        description = DescribeMap(
            mapu, prop_update, instrs, turn_state, follower, leader
        )
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(self.turn_number, action_number,"\n",description)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

        if instrs[-1].text != self.last_instruction:
            self.turn_number += 1
        self.instruction_parts.append(f"\n instruction:{instrs[-1].text}\nyour point of view:")
        game.visualization().visualize_follower_visibility(self.turn_number, action_number)
        image_data = Path(crop_non_white_square(
            f"follower_view/follower_visibility_{self.turn_number}_{action_number}.png")).read_bytes()
        self.image_parts.append(
            {
                "mime_type": "image/png",
                "data": image_data
            }
        )
        self.last_instruction = instrs[-1].text

        prop_update = PropUpdate(props)
        (leader, follower) = get_actors(game_state)

        self.messages = [self.game_explanation+self.instruction_parts[-1], self.image_parts[-1], '\n']
        response = call_gemini_api_sync(messages=self.messages,
                                        model=self.model)
        if not response:
            return Action.NoopAction()
        response_text = response.text
        action_string = ""

        lines = response_text.split("\n")
        for line in lines:
            if line.startswith("ACTIONS:") or line.startswith("ACTION:"):
                action_string = line.split(":")[1]
                # Strip punctuation from the end of the action string.
                action_string = action_string.rstrip(string.punctuation)
            else:
                # Make sure the line isn't just whitespace/punctuation.
                if line.strip(string.whitespace + string.punctuation) == "":
                    continue
                # If the line doesn't start with THOUGHTS or ACTIONS, this is
                # unexpected. GPT is probably not working as expected.

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
    response = model.generate_content(messages) #google.generativeai.GenerativeModel.generate_content
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
