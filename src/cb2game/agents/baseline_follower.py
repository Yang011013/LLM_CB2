from cb2game.agents.agent_utils import *
from random import gauss
from time import sleep, time
from cb2game.pyclient.follower_data_masking import (
    CensorActors,
    CensorFollowerMap,
    CensorFollowerProps,
)
import follower_bots.constants as const
import torch
from follower_bots.data_utils.data_classes import ActionEnums
from follower_bots.data_utils.pyclient_utils import (
    follower_idx_to_game_action,
    generate_action_mask,
    get_active_uuid,
    get_processed_actions,
    get_processed_instructions,
    get_processed_states,
)
from follower_bots.models.model_utils import load_follower_model_for_corpora_eval
class Config():
    fog_end = 20
import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass

from mashumaro.mixins.json import DataClassJSONMixin

from cb2game.agents.agent import Agent, Role
from cb2game.pyclient.game_endpoint import Action, GameState
@dataclass
class BaselineFollowerConfig(DataClassJSONMixin):
    experiments_folder: str
    experiments_name: str
    use_ensembling: bool

class BaselineFollower(Agent):
    def __init__(self, config: BaselineFollowerConfig) -> object:
        super().__init__()
        self.instructions_processed = set()
        self.model_name = "baseline_follower"
        self.actions = []
        self.config = config
        self.model = load_follower_model_for_corpora_eval(self.config)
        self.total_timesteps = 0
        self.states, self.actions, self.timesteps = [], [], []
        self.server_config = Config()
    # OVERRIDES role
    def role(self) -> Role:
        return Role.FOLLOWER
    # OVERRIDES choose_action
    def choose_action(self, game_state: GameState, action_mask=None, follower_view_path=None, test=False) -> Action:
        raw_instruction, proc_instruction, text_mask = None, None, None

        map, cards, turn_state, instructions, actors, feedback = game_state # 全局游戏的game_state
        (leader, follower) = get_actors(game_state)
        map = CensorFollowerMap(map, follower, self.server_config)
        cards = CensorFollowerProps(cards, follower, self.server_config)

        raw_instruction, proc_instruction, text_mask = get_processed_instructions(
            instructions, raw_instruction, proc_instruction, text_mask
        )
        terminated_instruction = raw_instruction.uuid != get_active_uuid(
            instructions
        )
        if terminated_instruction:
            self.states, self.actions, self.timesteps = [], [], []
            self.model.reset_past_output()
            self.total_timesteps = 0
        start_time = time()
        self.actions = get_processed_actions(self.actions)
        self.states = get_processed_states(self.states, map, cards, actors)
        self.timesteps = torch.LongTensor([[i for i in range(self.states.shape[1])]])
        self.attention_mask = torch.ones(*self.states.shape[:2], dtype=torch.long)
        pos_idx = torch.arange(
            0, proc_instruction.shape[1] + 2 * self.timesteps.shape[1], dtype=torch.long
        ).unsqueeze(0)

        if self.total_timesteps < const.INFERENCE_HORIZON:
            action_mask = generate_action_mask(action_mask)
            with torch.no_grad():
                action = self.model.sample_action(
                    self.states,
                    self.actions,
                    self.timesteps,
                    proc_instruction,
                    pos_idx,
                    self.attention_mask,
                    text_mask,
                    action_mask,
                )
        else:
            action = ActionEnums["DONE"].value
        self.actions[:, -1] = action
        game_action = follower_idx_to_game_action(action, raw_instruction.uuid)

        inference_time = time() - start_time
        print("inference time: ", inference_time)
        # map, cards, turn_state, instructions, actors, feedback = game.step(game_action)
        self.total_timesteps += 1

        # Reset states if instruction terminated or done
        done_instruction = (
            game_action.action_code() == Action.ActionCode.INSTRUCTION_DONE
        )
        terminated_instruction = raw_instruction.uuid != get_active_uuid(
            instructions
        )
        if done_instruction:
            self.states, self.actions, self.timesteps = [], [], []
            self.model.reset_past_output()
            self.total_timesteps = 0

        if terminated_instruction:
            self.states, self.actions, self.timesteps = [], [], []
            self.model.reset_past_output()
            self.total_timesteps = 0
        time_beyond_standard = max(0, inference_time - 0.15)
        sleep_time = max(0.1, gauss(0.7 - time_beyond_standard, 0.08))
        sleep(sleep_time)
        return "", game_action
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