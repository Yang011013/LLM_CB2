from typing import Union, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import google.generativeai as genai
from mashumaro.mixins.json import DataClassJSONMixin

from cb2game.agents.agent import Agent, RateLimitException, Role
from cb2game.pyclient.client_utils import SystemPrompt
from cb2game.agents.agent_utils import *
from cb2game.util.log_config import logger
import json
class Config():
    fog_end = 20


@dataclass
class GeminiFollowerConfigAtomic(DataClassJSONMixin):
    """Configuration for initializing a GeminiVisionFollowerAtomic agent."""
    gemini_api_key: str
    model: str = "gemini-1.5-pro-latest"
    temperature: float = 0.4
    top_p: int = 1
    top_k: int = 32
    queueing_enabled: bool = True
    safety_settings: list = None


class GeminiFollowerAtomic(Agent):
    def __init__(self, config: GeminiFollowerConfigAtomic):
        genai.configure(api_key=config.gemini_api_key)
        self.server_config = Config()
        self.gemini_config = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
        }

        self.model_name = config.model or "gemini-1.5-flash-latest"
        self.model = genai.GenerativeModel(model_name=self.model_name,
                                           generation_config=self.gemini_config,
                                           safety_settings=config.safety_settings)
        self.system_prompt = SystemPrompt()
        self.current_instruction = None
        self.deferred_task = ""
        self.last_deferred_task = ""
        self.action_queue = []

    def role(self) -> Role:
        return Role.FOLLOWER

    # OVERRIDES choose_action
    def choose_action(self, game_state: GameState, game=None, action_number=None, action_mask=None,
                    follower_view_path=None, test=False) -> Union[tuple[str, Any, Any], tuple[Any, Any, Any]]:
        image_file_path = follower_view_path or "follower_view/follower_first_view_play.png"
        
        mapu, props, turn_state, instrs, actors, *_ = game_state
        
        active_instruction = get_active_instruction(instrs)
        if active_instruction is None:
            return "", Action.NoopAction()

        if instrs[-1].uuid != self.current_instruction and test:
            self.action_queue.clear()

        if self.action_queue and self.queueing_enabled:
            return "", self.action_queue.pop(0)

        leader, follower = (None, actors[0]) if len(actors) == 1 else actors

        # Fetch prompt and update if instruction is new
        is_new_instruction = instrs[-1].uuid != self.current_instruction
        instruction_text = instrs[-1].text if is_new_instruction else self.deferred_task
        prop_update, prompt, map_description = get_prompt(instruction_text, mapu, props, follower, self.server_config, image_file_path)

        if is_new_instruction:
            logger.info(f"New instruction: {instruction_text}")
            self.current_instruction = instrs[-1].uuid
            self.last_deferred_task = ""
        
        self.last_format_err = ""

        # Retry until the response passes the format check
        while True:
            response = call_gemini_api_sync(messages=[self.system_prompt] + prompt, model=self.model)
            if not response:
                return "", Action.NoopAction()

            response_text = response.text
            response_dict, format_check = format_checker(response_text, image_file_path)
            logger.info(f"Response: {response_dict}\nFormat check: {format_check}")

            # Return if format error repeats
            if self.last_format_err == format_check:
                return "", Action.InstructionDone(active_instruction.uuid)

            if format_check is None:
                break

            self.last_format_err = format_check
            prompt += f"Your last response:\n{response_text} had the following error:\n{format_check}.\nPlease correct it:"

        self.deferred_task = response_dict.get("Deferred Task", "NULL")
        if self.deferred_task == self.last_deferred_task and self.deferred_task != "NULL":
            action_string = "done"
            self.last_deferred_task = ""
        else:
            self.last_deferred_task = self.deferred_task
            action_string = get_action_string(response_dict, mapu, prop_update, follower, image_file_path)

        logger.info(f"Action string: {action_string}")

        actions = actions_from_code(action_string, active_instruction.uuid)
        if not actions:
            return "", Action.NoopAction()
        
        if self.queueing_enabled:
            self.action_queue = actions[1:]
        return response_text, actions[0]

# @timeout_decorator(timeout=120)
@delay_execution(delay_time=10)
def call_gemini_api_sync(messages, model):
    response = model.generate_content(contents=messages)
    return response

def get_active_instruction(instructions):
    for instruction in instructions:
        if not instruction.completed and not instruction.cancelled:
            return instruction
    return None