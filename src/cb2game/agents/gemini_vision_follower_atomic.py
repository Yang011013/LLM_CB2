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
from cb2game.server.messages.prop import PropUpdate
from cb2game.server.db_tools import follower_view_description
from cb2game.pyclient.follower_data_masking import (
    CensorActors,
    CensorFollowerMap,
    CensorFollowerProps,
)
from cb2game.agents.agent_utils import *

logger = logging.getLogger(__name__)
class Config():
    fog_end = 20
@dataclass
class GeminiVisionFollowerConfigAtomic(DataClassJSONMixin):
    """Configuration for initializing a GeminiVisionFollowerAtomic agent."""
    gemini_api_key: str
    model: str = "gemini-1.5-pro-latest"# gemini-1.0-vision-pro
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
        print("========model_name========: ",self.model_name)
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
        self.system_prompt = SystemPrompt()
        self.game_history = []
        self.current_instruction = ''
        self.image_parts = []
        self.deferred_task = ""

        self.action_queue = []
        self.turn_number = 0

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
    def choose_action(self, game_state: GameState, game=None, action_number=None, action_mask=None) -> Action:
        if len(self.action_queue) > 0 and self.queueing_enabled:
            return self.action_queue.pop(0)
        # Fetch more actions.
        [mapu, props, turn_state, instrs, _, _] = game_state # 5.13这里返回是整个地图的mapu
        (leader, follower) = get_actors(game_state)
        follower_map_update = CensorFollowerMap(mapu, follower, self.server_config)

        follower_props = CensorFollowerProps(props, follower, self.server_config)
        prop_update = PropUpdate(follower_props)

        description = DescribeMap(follower_map_update, prop_update, instrs, turn_state, follower, leader)
        # 保存description到text文件
        with open("description.txt", "w") as f:
            f.write(description)

        # 保存第一视角图片
        # game.visualization().visualize_follower_visibility(self.turn_number, action_number)
        game.visualization().visualize_follower_visibility(0, 0)
        first_view_image_path = crop_non_white_square(f"follower_view/follower_visibility_0_0.png")
        image_data = Path(first_view_image_path).read_bytes()
        self.image_parts.append({"mime_type": "image/png", "data": image_data})

        if instrs[-1].text != self.current_instruction:
            print("=========================================================")
            print("instruction: ",instrs[-1].text)
            print("=========================================================")
            p1 = f"""
            Here in the instruction you received from the leader: {instrs[-1].text}
            the corresponding first-view png image: \n
            """
            p2 = f"Here is the structured string representing your first-view map: \n{description}"
            p3 = "Please provide your response:\n"
            prompt = [p1, self.image_parts[-1], p2, p3]  
            self.current_instruction = instrs[-1].text
            # 每得到一个新的指令重置记忆
            self.game_history = []
            self.turn_number += 1
        else:
            p1 = f"""
            Here in the new instruction you received from the leader: {self.deferred_task}
            the new corresponding first-view png image: \n
            """
            p2 = f"Here is the new structured string representing your first-view map: \n{description}"
            p3 = "Please provide your response:\n"
            prompt = [p1, self.image_parts[-1], p2, p3] 
        with open("game_history.txt", "w") as f:
            f.write(str(self.game_history))
        self.game_history = prompt
        response = call_gemini_api_sync(messages=[self.system_prompt] + self.game_history,
                                        model=self.model)
        

        if not response:
            self.game_history += ["model: Do nothing!"]
            return Action.NoopAction()
        else:
            response_text = response.text
            self.game_history += [f"{response_text}"]
        action_string = ""
        if self.model_name == "gemini-1.5-pro-latest":
            start_index = response_text.find("{") 
            end_index = response_text.find("}")
            response_text = response_text[start_index:end_index+1]

        print("response:\n",response_text)
        response_dict = json.loads(response_text)
        self.deferred_task = response_dict["Deferred Task"]

        action_string = get_action_string(response_dict, mapu, prop_update, follower)
        active_instruction = get_active_instruction(instrs)
        actions = actions_from_code(action_string, active_instruction.uuid)

        if len(actions) == 0:
            return Action.NoopAction()
        if self.queueing_enabled:
            self.action_queue = actions[1:]
            return actions[0]

        return actions[0]


# @timeout_decorator(timeout=120)
# @delay_execution(delay_time=10)
def call_gemini_api_sync(messages, model):
    response = model.generate_content(messages)
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
