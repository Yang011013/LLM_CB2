from typing import Union, Tuple, Any
from dataclasses import dataclass
import ollama
from openai import OpenAI
from mashumaro.mixins.json import DataClassJSONMixin

from cb2game.agents.agent import Agent, RateLimitException, Role
from cb2game.pyclient.client_utils import SystemPrompt
from cb2game.agents.agent_utils import *
from cb2game.util.log_config import logger
from unsloth import FastLanguageModel 
import re
import random

import os
os.environ['CURL_CA_BUNDLE'] = ''

class Config():
    fog_end = 20


@dataclass
class ModelConfig(DataClassJSONMixin):
    model: str  # model name
    queueing_enabled: bool
    max_seq_length: int

class Models(Agent):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.server_config = Config()
        self.model_name = config.model
        self.queueing_enabled = config.queueing_enabled

        self.client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama')

        with open("src/cb2game/agents/v2instr.txt", 'r') as f:
            self.v2_prompt = f.read()
        self.Instruction = "\nYou would be provided with a instrction and an structured string to describe your first-view map.\nYour task is to break down the original instruction into two categories based on the map:\n1. Immediate Tasks: Tasks that are achievable within your current perspective and can be completed promptly. Type 1: Change Direction (e.g., \"Turn left/right/Around\") Type 2: Move to a specific location in the first-view map(e.g., \"Next Location: Tile at heading <angle> and distance <distance>: <GROUND_TILE_TYPE>\"). Type 3: Interact with a Card at a Specific Location within the visible area of the map.\n2. Deferred Tasks: Tasks that necessitate a change in perspective or additional insights to be accomplished. If there are no deferred tasks, record the output as \"NULL\".\n\nProvide your answer in JSON format with the following keys:Immediate Task, Deferred Task. Other formats are not accepted.\nExpected Output Format:\n{\"Immediate Task\": Either a direction change (e.g., \"Turn left/right/Around\") or a specific location (e.g., \"Next Location: Tile at heading <angle> and distance <distance>: <GROUND_TILE_TYPE>\",\"Card Interaction:   <interaction>  at Tile at heading <angle> and distance <distance>: <GROUND_TILE_TYPE>\"),\n\"Deferred Task\": \"NULL\" or a consice description of the remaining instructions in no more then 20 words.}\n\n"
        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""
        self.current_instruction = None
        self.deferred_task = ""
        self.last_deferred_task = ""
        self.action_queue = []



        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = f"models/{self.model_name}/",
            max_seq_length = config.max_seq_length,
            dtype = None,
            load_in_4bit = True,
        )
        self.E0S_TOKEN = self.tokenizer.eos_token
        FastLanguageModel.for_inference(self.model)
        self.format_check = True

    def role(self) -> Role:
        return Role.FOLLOWER

    # OVERRIDES choose_action
    def choose_action(self, game_state: GameState, game=None, action_number=None, action_mask=None,
                    follower_view_path=None, test=False) -> Union[tuple[str, Any, Any], tuple[Any, Any, Any]]:
        
        mapu, props, turn_state, instrs, actors, *_ = game_state

        active_instruction = get_active_instruction(instrs)
        if active_instruction is None:
            return "", Action.NoopAction()
        # testing model: when get new instruction, clear the actions queue
        if instrs[-1].uuid != self.current_instruction and test:
            self.action_queue = []

        if len(self.action_queue) > 0 and self.queueing_enabled:
            return "", self.action_queue.pop(0)
        (leader, follower) = (None, actors[0]) if len(actors) == 1 else actors

        # Fetch prompt and update if instruction is new
        is_new_instruction = instrs[-1].uuid != self.current_instruction
        instruction_text = instrs[-1].text if is_new_instruction else self.deferred_task

        if is_new_instruction:
            logger.info(f"New instruction: {instruction_text}")
            self.current_instruction = instrs[-1].uuid
            self.last_deferred_task = ""
        prop_update, _, map_description = get_prompt(instruction_text, mapu, props, follower, self.server_config)
        input_str = f"Instruction: {instrs[-1].text}, \nFirst-view Map:MAP DIMENSIONS: {map_description}"
        self.last_format_err = ""

        
        prompt = self.alpaca_prompt.format(
                    self.Instruction, # if you want to use the v2_prompt, change it with `self.system_prompt`
                    input_str,
                    "", # output - leave this blank for generation!
                )       
        while True:
            inputs = self.tokenizer(
                [
                    prompt
                ], return_tensors = "pt").to("cuda")
            outputs = self.model.generate(**inputs, max_new_tokens = 64, use_cache = True)
            output_lst = self.tokenizer.batch_decode(outputs)
            output_text = output_lst[0]


            s_idx = output_text.find("### Response:\n") + len("### Response:\n")
            e_idx = output_text.find(self.E0S_TOKEN)
            response = output_text[s_idx:e_idx]        

            if not response:
                return "", Action.NoopAction()
            response_dict, format_check = format_checker(response, map_description)

            logger.info(f"\nresponse_dict:\n{response_dict}\nformat_check:\n{format_check}")

            if self.last_format_err == format_check:
                self.last_format_err = ""      
                break
            else:
                response_text = response
            if format_check is None:
                break
            else:
                self.last_format_err = format_check
                format_error_prompt = f"Your last response:\n{response_text} with the following kind of error:\n{format_check}.\nPlease give another correct response:\n"
        
        if format_check is not None:
            if format_check in ["When you deselect a card, the location of the card should be extracted from the SELECTED CARDS part of the structured string of your first-view map provided.", "When you select a card, the location of the card should be extracted from the UNSELECTED CARDS part of the structured string of your first-view map provided."]:
                response_dict = random_response(response_dict, format_check, map_description)
            else:
                return "", Action.InstructionDone(active_instruction.uuid)

        self.deferred_task = response_dict["Deferred Task"]
        # if LLM keep response the same deferred task
        if self.last_deferred_task == self.deferred_task and self.deferred_task != "NULL":
            self.last_deferred_task = ""
            action_string = "done"
        else:
            self.last_deferred_task = self.deferred_task
            action_string = get_action_string(response_dict, mapu, prop_update, follower)
        logger.info(f"action string: {action_string}")

        actions = actions_from_code(action_string, active_instruction.uuid)

        if len(actions) == 0:
            return "", Action.NoopAction()
        if self.queueing_enabled:
            self.action_queue = actions[1:]

        return response_text, actions[0]

def random_response(response_dict, format_check, map_description):
    selected_cards_pattern = r'SELECTED CARDS:\s*(.*?)\s*UNSELECTED CARDS:'
    unselected_cards_pattern = r'UNSELECTED CARDS:\s*(.*?)\s*MAP DESCRIPTION'

    selected_cards_match = re.search(selected_cards_pattern, map_description, re.DOTALL)
    selected_cards = selected_cards_match.group(1).strip() if selected_cards_match else "No cards found"
    selected_cards = [x.strip() for x in selected_cards.split("\n") if x.strip()]

    unselected_cards_match = re.search(unselected_cards_pattern, map_description, re.DOTALL)
    unselected_cards = unselected_cards_match.group(1).strip() if unselected_cards_match else "No cards found"
    unselected_cards = [x.strip() for x in unselected_cards.split("\n") if x.strip()]
    try:
        if format_check == "When you deselect a card, the location of the card should be extracted from the SELECTED CARDS part of the structured string of your first-view map provided.":
            response_dict["Immediate Task"] = f"Card Interaction: ['Select Card at {random.choice(selected_cards)}']"
        elif format_check == "When you select a card, the location of the card should be extracted from the UNSELECTED CARDS part of the structured string of your first-view map provided.":
            response_dict["Immediate Task"] = f"Card Interaction: ['Deselect Card at {random.choice(unselected_cards)}']"
    except:
        return None
    if "No cards found" in response_dict["Immediate Task"]:
        return None
    else:
        return response_dict

def get_active_instruction(instructions):
    for instruction in instructions:
        if not instruction.completed and not instruction.cancelled:
            return instruction
    return None
