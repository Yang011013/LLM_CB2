"""This is a template file for creating a new Agent.

Replace the class and method definitions with your own implementations.
"""
import os
from typing import Union, Tuple, Any

from cb2game.server.messages.action import Action

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
import PIL
import json
import time
import tiktoken
from mashumaro.mixins.json import DataClassJSONMixin

from cb2game.agents.agent import Agent, RateLimitException, Role
from cb2game.pyclient.client_utils import (
    DescribeMap,
    FollowerSystemPrompt,
    SingleActionSystemPrompt,
    crop_non_white_square,
    SystemPrompt
)
from cb2game.pyclient.game_endpoint import Action, GameState

from cb2game.server.db_tools import follower_view_description

from cb2game.agents.agent_utils import *

logger = logging.getLogger(__name__)


class Config():
    fog_end = 20


@dataclass
class GeminiVisionFollowerConfigAtomic(DataClassJSONMixin):
    """Configuration for initializing a GeminiVisionFollowerAtomic agent."""
    gemini_api_key: str
    model: str = "gemini-1.5-pro-latest"  # gemini-1.0-vision-pro
    max_output_tokens: int = (
        4096  # Model maximum of 12288. Completion consumes some tokens too though.
    )
    max_in_tokens: int = 30000
    temperature: float = 0.4
    top_p: int = 1
    top_k: int = 32
    queueing_enabled: bool = False


class GeminiVisionFollowerAtomic(Agent):
    def __init__(self, config: GeminiVisionFollowerConfigAtomic):
        super().__init__()

        self.temperature = config["temperature"]
        self.top_p = config["top_p"]
        self.top_k = config["top_k"]
        genai.configure(api_key=config["gemini_api_key"])
        self.model_name = config["model"]
        self.max_output_tokens = config["max_output_tokens"]
        self.max_in_tokens = config["max_in_tokens"]
        self.queueing_enabled = config["queueing_enabled"]
        self.server_config = Config()
        print("========model_name========: ", self.model_name)
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
        if self.model_name is not None:
            self.model = genai.GenerativeModel(model_name=self.model_name,
                                               generation_config=self.gemini_config,
                                               safety_settings=self.safety_settings)
        else:
            self.model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                                               generation_config=self.gemini_config,
                                               safety_settings=self.safety_settings)
        # self.model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
        self.system_prompt = SystemPrompt()
        self.game_history = []
        self.current_instruction = ''
        self.image_parts = []
        self.deferred_task = ""
        self.last_deferred_task = ""

        self.action_queue = []

    def role(self) -> Role:
        return Role.FOLLOWER

    def add_to_game_history(self, role, parts):
        self.game_history.append({"role": role, "parts": parts})

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
    def choose_action(self, game_state: GameState, game=None, action_number=None, action_mask=None,
                      test=False) -> Union[tuple[str, Any], tuple[Any, Any]]:
        # Fetch more actions.
        [mapu, props, turn_state, instrs, _, _] = game_state  # 5.13这里返回是整个地图的mapu
        active_instruction = get_active_instruction(instrs)
        # testing model: when get new instruction, clear the actions queue
        if instrs[-1].text != self.current_instruction and test:
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            self.action_queue = []

        if len(self.action_queue) > 0 and self.queueing_enabled:
            action = self.action_queue.pop(0)
            return "", action
        (leader, follower) = get_actors(game_state)

        # save the first-view RGB map
        image_file_path = "follower_view/follower_first_view.png"
        game.visualization().visualize_follower_visibility(file_path=image_file_path)
        crop_non_white_square(image_file_path)
        self.image_parts.append({"mime_type": "image/png", "data": Path(image_file_path).read_bytes()})

        if instrs[-1].text != self.current_instruction:
            print("=============================================")
            print("instruction: ",instrs[-1].text)
            print("=============================================")
            prop_update, prompt = get_prompt(instrs[-1].text, mapu, props, follower, self.image_parts[-1], self.server_config)
            self.current_instruction = instrs[-1].text
            self.last_deferred_task = ""
            # when get new instruction, refresh the game_history
            self.game_history = []
        else:
            prop_update, prompt = get_prompt(self.deferred_task, mapu, props, follower, self.image_parts[-1], self.server_config)
        self.game_history = prompt
        self.last_format_err = ""
        while True:
            try:
                response = call_gemini_api_sync(messages=[self.system_prompt] + self.game_history,
                                                model=self.model)
            except Exception as e:
                time.sleep(3)
                response = call_gemini_api_sync(messages=[self.system_prompt] + self.game_history,
                                                model=self.model)
            if not response:
                self.game_history += ["model: Do nothing!"]
                return "", Action.NoopAction()
            else:
                response_text = response.text
                self.game_history += [f"{response_text}"]
            response_dict, format_check = format_checker(response.text)
            # If format_check is not None, response_dict will be an empty string
            print(f"response_dict:\n{response_dict}\nformat_check:\n{format_check}")
            # if repeat response the same format check error, response Done action
            if self.last_format_err == format_check:
                self.last_format_err = ""
                return "", Action.InstructionDone(active_instruction.uuid)
            if format_check is None:
                break
            else:
                self.last_format_err = format_check

                self.game_history += format_check


        self.deferred_task = response_dict["Deferred Task"]
        # if LLM keep response the same deferred task
        if self.last_deferred_task == self.deferred_task and self.last_deferred_task != "NULL":
            self.last_deferred_task = ""
            action_string = "done"
        else:
            self.last_deferred_task = self.deferred_task
            action_string = get_action_string(response_dict, mapu, prop_update, follower)

        actions = actions_from_code(action_string, active_instruction.uuid)

        if len(actions) == 0:
            return "", Action.NoopAction()
        if self.queueing_enabled:
            self.action_queue = actions[1:]
            return response_text, actions[0]

        return response_text, actions[0]





# @timeout_decorator(timeout=120)
@delay_execution(delay_time=1)
def call_gemini_api_sync(messages, model):
    with open("messages.txt", "w") as f:
        f.write(str(messages))
    response = model.generate_content(contents=messages)
    return response


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
