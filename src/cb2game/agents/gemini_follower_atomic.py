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
from cb2game.server.db_tools import follower_view_description

logger = logging.getLogger(__name__)

@dataclass
class GeminiFollowerConfigAtomic(DataClassJSONMixin):
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


class GeminiFollowerAtomic(Agent):
    def __init__(self, config: GeminiFollowerConfigAtomic):
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
        
        self.system_prompt = [
            {"role": "user", "parts": """
                You are an embodied agent collaborating with a human leader within an interactive environment. Your primary task is to perform actions based on the instructions provided by the leader. You have the capability to navigate through the environment by moving to adjacent hexagonal tiles and by pivoting to adjust your orientation. Additionally, you can interact with cards by moving over them to select or deselect them. 

                While the leader has access to a comprehensive overhead view of the entire environment, your perspective is limited to a first-person view directly in front of you. You cannot see things behind you or to the sides. You also can't see things that are too far away. This difference in vantage points may result in a vision gap, where the leader's instructions might not be immediately clear from your current perspective. To address this, you may need to take exploratory actions, such as moving forward or turning, to gain a new view and better understand the leader's instructions. 

                In this environment, you can perform the following automated actions:
                - Turn Right: Rotate your direction to the right without moving from the current tile.
                - Turn Left: Rotate your direction to the left without moving from the current tile.
                - Move Forward: Advance one step in your current direction while maintaining the same orientation.
                - Move Back: Retreat one step in the opposite direction of your current orientation, without changing your facing direction.
                - Done: Indicate that you have completed the current instruction from the leader. Or the leader's command is "wait" or other commands that you don't need to take any action on. 
                    
                You can perform a turn back(turn around) by turn right or left three times, as this is an environment made up of hexagonal tiles.
                If the house/lake/tree/streetlight/rock one tile in front of you or one tile behind you, then you cannot choose move forward or backward. 

                Standing on the card means you select/get the card, stepping away from it and then standing on the card again means you deselect the card.

                About Structured String to describe your first-view map:\n
                This environment consists of a hexagonal tile grid, allowing each player six possible orientations. \n
                Due to limited visibility, your perspective is restricted to a fan-shaped area directly in front of you. \n
                The environment's first-person view can be represented by a structured string with the following components:

                - MAP DIMENSIONS: A {map.rows}x{map.cols} hexagonal grid featuring {num_cards_in_view} interactive elements.
                - CARDS DESCRIPTION: Instructions on how to navigate to the tile containing a card.
                - MAP DESCRIPTION: Descriptions of the lake, mountain, city, and leader visible within the view.
                - NEARBY TILES:
                    - Left Tile: At a heading of 60° and a distance of 1.0 unit, named {AssetId(tile.asset_id).name}.
                    - Right Tile: At a heading of -60° and a distance of 1.0 unit, named {AssetId(tile.asset_id).name}.
                    - Forward Tile: Directly ahead at a heading of 0° and a distance of 1.0 unit, named {AssetId(tile.asset_id).name}.
                (Note: The tile behind you is not visible.)
                - FURTHER TILES: A list of tiles beyond immediate view, formatted as "Tile at heading {direction:.0f}° and distance {distance:.1f} units: {AssetId(tile.asset_id).name}."

                How to break down the instructions:
                Objective: Interpret the leader's instructions by exploring the environment to continuously gain new perspectives. You need to categorize these instructions into two parts: immediate tasks and deferred tasks.

                Immediate Task: These are tasks that can be completed within your current field of view. \n
                There are two types of immediate tasks:
                Type 1: Change Direction: Choose one of the following:
                - Turn Left
                - Turn Right
                - Turn Around
                Type 2: Finish the Instruction: Indicate that you have completed the current instruction from the leader. Or the leader's command is "wait" or other commands that you don't need to take any action on.
                - Done
                Type 3: Move to a Specific Location: Navigate within the visible area of the map. Specifically, interact with items by standing on a card to select it. To deselect a card, step away from it and then stand on it again.
                The type of the tile you chose can only be limited to GROUND_TILE_PATH, GROUND_TILE, RAMP_TO_MOUNTAIN, MOUNTAIN_TILE and CARD.  Describe the location with the format: Tile at heading  <angle> and distance <distance>: <TILE_TYPE>.You should decide which tile in your first-view map to go to based on the instructions.
                If the leader instructs you to pick a card, you should check whether the card is in your view. If it is, you can directly point out the card's location. If the card is not in your view, you should change direction or move to a new location to find it.

                Deferred Task: These are tasks that require a change in perspective, which becomes possible after completing the immediate tasks. Describe these tasks in natural language.

                Please note that the 'Immediate Task' is not necessarily the first part of the instruction, and the 'Deferred Task' does not only refer to the rest half of the instruction. This requires you to decompose the 'Immediate Task' and the 'Deferred Task' based on the complete instruction and your current first-person perspective.

                Process for Breaking Down Instructions:

                1、 Analyze the View: Identify the tasks that can be performed immediately based on your current first-view map.
                2、Identify Deferred Tasks: Summarize any remaining tasks that cannot be completed yet as deferred tasks. If there are no such tasks, record the output for this section as "NULL".
                3、Review and Adjust:
                - Ensure that your immediate and deferred tasks accurately represent the leader’s instructions. Adjust them if necessary to capture all details.
                - Ensure the location you describe for immediate tasks is accurate based on the structured string provided. If not, choose the correct location and update the description accordingly.

                Your response should be formatted as:
                'Immediate Task': Either a direction change (e.g., "Turn left/right/Around") or Finish the instruction(i.e. Done) or a specific location (e.g., "Next Location: Tile at heading <angle> and distance <distance>: <GROUND_TILE_TYPE>"). Please note that you can only response in one of these two formats, other content is not allowed
                'Deferred Task': Either "NULL" or The unfinished part of the instruction.

                At the beginning of each turn, you can get instructions from the leader and your first-view map. After that, I will execute the 'Immediate Task' of your response each time on the game terminal and provide you with a new first-view map.
                """},
            {"role": "model", "parts": "Okay, I already know the rules."},
        ] 
        self.game_history = []
        self.current_instruction = ''

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
    def choose_action(self, game_state: GameState, action_mask=None) -> Action:
        if len(self.action_queue) > 0 and self.queueing_enabled:
            return self.action_queue.pop(0)
        # Fetch more actions.
        [mapu, props, turn_state, instrs, _, _] = game_state
        prop_update = PropUpdate(props)
        (leader, follower) = get_actors(game_state)

        description_atomic = follower_view_description.DescribeMap(mapu, prop_update.props, instrs, follower.location(), follower.heading_degrees())
        with open("description_atomic.txt", "w") as f:
            f.write(description_atomic)
        description = DescribeMap(mapu, prop_update, instrs, turn_state, follower, leader)
        # 保存description到text文件
        with open("description.txt", "w") as f:
            f.write(description)
        
        if instrs[-1].text != self.current_instruction:
            prompt = f"Here in the instruction you received from the leader: {instrs[-1].text}\nHere is the structured string representing your first-view map: \n{description}\nPlease provide your response:\n"
            self.current_instruction = instrs[-1].text
            # 每得到一个新的指令重置记忆
            self.game_history = []
        else:
            prompt = f"Here is the new structured string representing your first-view map: \n{description}\nPlease provide your response:\n"

        response = call_gemini_api_sync(messages=self.system_prompt + self.game_history,
                                        instruction = prompt,
                                        model=self.model)
        self.add_to_game_history("user", prompt)

        if not response:
            self.add_to_game_history("model", "Do nothing!")
            return Action.NoopAction()
        else:
            response_text = response.text
            self.add_to_game_history("model", response_text)
        action_string = ""

        with open("game_history.txt", "w") as f:
            for message in self.system_prompt:
                for val in message.values():
                    f.write(val)
            for message in self.game_history:
                for val in message.values():
                    f.write(val)
        print("response:\n",response_text)

        lines = response_text.split("\n")
        for line in lines:
            if 'Immediate Task' in line:
                if "turn" or "done" in line.lower():
                    action_string = line.split(":")[1].strip()
                elif "next location" in line.lower():
                    # line=Immediate Task: Next Location: Tile at heading  <angle> and distance <distance>: <TILE_TYPE>
                    action_string = find_matching_tiles(description_atomic, line.split("Next Location:")[1].strip())
                else:
                    action_string = ""
            elif 'Deferred Task' in line:
                if "NULL" in line:
                    action_string += ", d"
            else:
                # Make sure the line isn't just whitespace/punctuation.
                if line.strip(string.whitespace + string.punctuation) == "":
                    continue
        print("action_string\n",action_string)
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
def call_gemini_api_sync(messages, instruction, model):

    conversation = model.start_chat(history=messages)
    response = conversation.send_message(instruction)

    return response

# def find_matching_tiles(data, keyword):# keyword-->AssetID
#     lines = data.split('\n')
#     matching_lines = []
#     found = False
#     for i, line in enumerate(lines):
#         if keyword in line:
#             found = True
#             for j in range(i+2, i+9):
#                 if j < len(lines) and "cannot reach" not in lines[j]:
#                     matching_lines.append(lines[j].split(":")[1] + ", d")
#                     break
#     if len(matching_lines) == 0:
#         matching_lines.append("d")
def find_matching_tiles(data, keyword):# keyword-->Next Location: Tile at heading  <angle> and distance <distance>: <TILE_TYPE>
    lines = data.split('\n')
    matching_line = ""
    found = False
    for i, line in enumerate(lines):
        if keyword in line:
            for j in range(i+2, i+9):
                if j < len(lines) and "cannot reach" not in lines[j]:
                    matching_line = lines[j].split(":")[1]
                    print("matching_line\n",matching_line)
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
        elif "backward".startswith(c) or "b".startswith(c):
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
