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
    FollowerSystemPrompt,
    SingleActionSystemPrompt,
)
from cb2game.pyclient.game_endpoint import Action, GameState
from cb2game.server.messages.prop import PropUpdate

logger = logging.getLogger(__name__)

@dataclass
class GeminiFollowerConfig(DataClassJSONMixin):
    """Configuration for initializing a GeminiFollower agent."""
    gemini_api_key: str
    model: str = "gemini-1.0-pro"
    max_output_tokens: int = (
        4096  # Model maximum of 12288. Completion consumes some tokens too though.
    )
    max_in_tokens: int = 30000
    temperature: float = 0.4
    top_p: int = 1
    top_k: int = 32
    queueing_enabled: bool = False


class GeminiFollower(Agent):
    def __init__(self, config: GeminiFollowerConfig):
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
        self.gemini_config = {
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "top_k": config["top_k"],
            "max_output_tokens": config["max_output_tokens"],
        }
        self.model = genai.GenerativeModel(model_name=self.model_name,
                                           generation_config=self.gemini_config,
                                           safety_settings=self.safety_settings)


        game_explanation = ( # 如果self.queueing_enabled为True，
            FollowerSystemPrompt()
            if self.queueing_enabled
            else SingleActionSystemPrompt()
        )
        self.game_history = [
            {"role": "user", "parts": game_explanation},
            {"role": "model", "parts": "Okay, I already know the rules."},
        ]
        self.action_queue = []

    def role(self) -> Role:
        return Role.FOLLOWER

    def add_to_game_history(self, role, parts):
        self.game_history.append({"role": role, "parts": [parts]})

    def _count_tokens(self, text: str):
        tokens = self.model.count_tokens(text)
        return tokens.total_tokens

    def _can_history_fit(self):
        tokens = sum([self._count_tokens(x["parts"]) for x in self.game_history])
        return tokens <= self.max_in_tokens

    def _prune_oldest_messages(self) -> bool:
        """Prunes the oldest non-system message in the history. Returns true if message removed."""
        del self.game_history[2:6]
        return True

    # OVERRIDES choose_action
    def choose_action(self, game_state: GameState, action_mask=None) -> Action:
        if len(self.action_queue) > 0 and self.queueing_enabled:
            return self.action_queue.pop(0)
        # Fetch more actions.
        [mapu, props, turn_state, instrs, _, _] = game_state
        prop_update = PropUpdate(props)
        (leader, follower) = get_actors(game_state)
        description = DescribeMap(
            mapu, prop_update, instrs, turn_state, follower, leader
        )

        self.add_to_game_history("user", description)
        self.add_to_game_history("model", "Okay, I already know the map information.")

        # make the game history fit
        # while not self._can_history_fit():
        #     self._prune_oldest_messages()


        response = call_gemini_api_sync(messages=self.game_history,
                                        model=self.model)

        self.add_to_game_history("user", "Now based on rules and your first-person view observation output:")

        if not response:
            self.add_to_game_history("model", "Do nothing!")
            return Action.NoopAction()

        else:
            response_text = response.text
            self.add_to_game_history("model", response_text)
        action_string = ""

        # Split by lines. If a line starts with "THOUGHTS:" or "THOUGHT:", then
        # print the line. If a line starts with "ACTIONS:" or "ACTION:", then
        # collect everything after the colon and split by comma.
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

    conversation = model.start_chat(history=messages)
    response = conversation.send_message("Now based on rules and information output:")

    with open("src/cb2game/agents/prompts/game_history.txt", 'w') as f:
        for message in messages:
            f.write(str(message))
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
