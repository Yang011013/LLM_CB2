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
        # image_data = PIL.Image.open(
        #     crop_non_white_square(f"follower_view/follower_visibility_{self.turn_number}_{action_number}.png"))
        # image_data = Path(crop_non_white_square(
        #     f"follower_view/follower_visibility_{self.turn_number}_{action_number}.png")).read_bytes()
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
        immediate_task = response_dict["Immediate Task"]
        # try: # 可能输出不规范
        #     response_dict = json.loads(response_text)
        #     response_dict = json.loads(response_text)
        #     self.deferred_task = response_dict["Deferred Task"]
        #     immediate_task = response_dict["Immediate Task"]
        # except KeyError as e:


        # 只要去具体位置才广搜
        if "Card Interaction" in immediate_task or "Next Location" in immediate_task: # Type1: Change Direction
            start_time = time.time()
            print("search tiles length: ",len(mapu.tiles))
            description_atomic = follower_view_description.DescribeMap(mapu, prop_update.props, instrs, follower.location(), follower.heading_degrees())
            end_time = time.time()
            print("Time taken to search map: ", end_time-start_time)
            with open("description_atomic.txt", "w") as f:
                f.write(description_atomic)

        if "Change Direction" in immediate_task: # Type1: Change Direction
            action_string = immediate_task.split(":")[1].strip()
        elif "Card Interaction" in immediate_task: #Type3: Next Location
            card_location = immediate_task.split("Card at")[1].strip()
            if "Deselect" in immediate_task:
                action_string = deselect_card(mapu, description_atomic, follower, card_location)
            elif "Select" in immediate_task:
                action_string = find_matching_tiles(description_atomic, card_location)

        elif "Next Location" in immediate_task: #Type3: Next Location
            action_string = find_matching_tiles(description_atomic, immediate_task.split("Next Location:")[1].strip())
        else:
            action_string = "done" 
        if self.deferred_task == "NULL":
            action_string += ",done"
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

def delay_execution(delay_time):
    def decorator(func):
        def wrapper(*args, **kwargs):
            time.sleep(delay_time)
            return func(*args, **kwargs)
        return wrapper
    return decorator
# @timeout_decorator(timeout=120)
# @delay_execution(delay_time=10)
def call_gemini_api_sync(messages, model):
    response = model.generate_content(messages)
    return response

def deselect_card(mapu, description_atomic, follower, card_location):
    # 1.卡片在当前位置：forward+backward+forward或者backward+forward+backward
    if "distance 0:" in card_location:
        # 还要考虑follower的朝向
        follower_orientation = follower.heading_degrees() - 60
        atomic_instructions = []
        # 检查周围的tile是否可行
        # 可行的tile.name: "GROUND_TILE":3, "GROUND_TILE_PATH":28,"MOUNTAIN_TILE":30,"RAMP_TO_MOUNTAIN":31,"SNOWY_MOUNTAIN_TILE":32,"SNOWY_RAMP_TO_MOUNTAIN":36
        actionable_tiles_id = [3, 28, 30, 31, 32, 36]
        for neighbors_location in follower.location().neighbors():
            # 检索该tile的名字，并判断是否在可行的tile中
            neighbors_tile_id = mapu.get_tile_id(neighbors_location)
            if neighbors_tile_id not in actionable_tiles_id:
                continue
            # 如果当前follower不在mountain类的tile上，那么只能选择GROUND_TILE和GROUND_TILE_PATH、RAMP_TO_MOUNTAIN、SNOWY_RAMP_TO_MOUNTAIN 
            if mapu.get_tile_id(follower.location()) in [3, 28]:
                if neighbors_tile_id not in [3, 28, 31, 36]:
                    continue
            # 如果当前follower在mountain类的tile上，那么只能选择MOUNTAIN_TILE、SNOWY_MOUNTAIN_TILE、RAMP_TO_MOUNTAIN、SNOWY_RAMP_TO_MOUNTAIN
            if mapu.get_tile_id(follower.location()) in [30, 32]:
                if neighbors_tile_id not in [30, 32, 31, 36]:
                    continue
            # 如果当前follower在RAMP上，志强选择""forward, backward, forward"或者"backward, forward, backward"
            if mapu.get_tile_id(follower.location()) in [31, 36]:
                # 先不考虑ramp的朝向
                atomic_instructions.append("forward, backward, forward")
                continue 
            degrees_away = follower.location().degrees_to(neighbors_location) - follower_orientation
            if degrees_away < 0:
                degrees_away += 360
            if degrees_away > 180:
                degrees_away -= 360
            if degrees_away == 0:
                atomic_instructions.append("forward, backward")
            elif degrees_away == 60:
                atomic_instructions.append("right, forward, backward")
            elif degrees_away == 120:
                atomic_instructions.append("left, backward, forward")
                #atomic_instructions.append("right, right, forward, backward")
            elif degrees_away == 180:
                atomic_instructions.append("backward, forward, backward")
            elif degrees_away == -60:
                atomic_instructions.append("left, forward, backward")
            elif degrees_away == -120:
                atomic_instructions.append("right, backward, forward")
                #atomic_instructions.append("left, left, forward, backward")
        return atomic_instructions[0] # 返回第一个可行的tile，的方式
    # 2.卡片不在当前位置：直接去卡片的位置
    else:
        atomic_instruction = find_matching_tiles(description_atomic, card_location)
        return atomic_instruction
    

def find_matching_tiles(data, keyword):# keyword-->Next Location: Tile at heading  <angle> and distance <distance>: <TILE_TYPE>
    if "distance 0.0" in keyword:
        return ""
    lines = data.split('\n')
    matching_line = ""
    found = False
    for i, line in enumerate(lines):
        if keyword.lower() in line.lower():
            if "You are standing here." in lines[i+2]:
                return ""
            for j in range(i+2, i+9):
                if j < len(lines) and "cannot reach" not in lines[j]:
                    matching_line = lines[j].split(":")[1]
                    break
            break
    return matching_line

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
        elif "forward" in c:
            actions.append(Action.Forwards())
        elif "backward".startswith(c) or "b".startswith(c):
            actions.append(Action.Backwards())
        elif "backward" in c:
            actions.append(Action.Backwards())
        elif "left".startswith(c) or "l".startswith(c):
            actions.append(Action.Left())
        elif "left" in c:
            actions.append(Action.Left())
        elif "right".startswith(c) or "r".startswith(c):
            actions.append(Action.Right())
        elif "right" in c:
            actions.append(Action.Right())
        elif "around" in c:
            actions.append(Action.Right())
            actions.append(Action.Right())
            actions.append(Action.Right())
        elif "done".startswith(c) or "d".startswith(c):
            actions.append(Action.InstructionDone(i_uuid))
        elif "noop" in c:
            actions.append(Action.NoopAction())
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
