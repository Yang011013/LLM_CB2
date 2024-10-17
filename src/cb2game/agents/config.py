""" Defines config for configuring CB2 agents. """
import dataclasses
import logging
from typing import Union

import yaml
from mashumaro.mixins.json import DataClassJSONMixin

from cb2game.agents.agent import Agent
from cb2_github.src.cb2game.agents.gemini_follower_atomic import GeminiFollowerAtomic, GeminiFollowerConfigAtomic
from cb2game.agents.gpt_follower_atomic import GPTFollowerAtomic,GPTFollowerConfigAtomic
from cb2game.agents.gpt_follower import GptFollower, GptFollowerConfig
from cb2game.agents.baseline_follower import BaselineFollower, BaselineFollowerConfig
from cb2_github.src.cb2game.agents.fintuned_mixtral import ModelConfig, Models
from cb2game.util.deprecated import deprecated

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AgentConfigData:
    name: str  # Name of the agent.
    comment: str  # Comment for the agent.
    agent_type: str  # Agent type. Contains module and class names.
    config: Union[
        GeminiFollowerConfigAtomic,
        BaselineFollowerConfig,
        GPTFollowerConfigAtomic,
        GptFollowerConfig,
        ModelConfig,
    ]  # Configuration for initializing the agent.


def ReadAgentConfigOrDie(config_path) -> AgentConfigData:
    with open(config_path, "r") as cfg_file:
        data = yaml.load(cfg_file, Loader=yaml.SafeLoader)
    
    if not isinstance(data, AgentConfigData):
        try:
            data = AgentConfigData(**data)
            
            # 根据 agent_type 动态构造具体的 config 对象
            if isinstance(data.config, dict):
                if data.agent_type == "GEMINI_VISION_FOLLOWER_ATOMIC":
                    data.config = GeminiFollowerConfigAtomic(**data.config)
                elif data.agent_type == "BASELINE_FOLLOWER":
                    data.config = BaselineFollowerConfig(**data.config)
                elif data.agent_type == "GPT_FOLLOWER":
                    data.config = GptFollowerConfig(**data.config)
                elif data.agent_type == "OLLAMA_MODELS":
                    data.config = ModelConfig(**data.config)
                else:
                    raise ValueError(f"Unknown agent_type: {data.agent_type}")
                    
        except TypeError as e:
            logger.error("Error parsing agent config: %s", e)
            raise

    return data



def LoadAgentFromConfig(config_data: AgentConfigData) -> Agent:
    if config_data.agent_type == "GEMINI_VISION_FOLLOWER_ATOMIC":
        return GeminiFollowerAtomic(config_data.config)
    elif config_data.agent_type == "BASELINE_FOLLOWER":
        return BaselineFollower(config_data.config)
    elif config_data.agent_type == "GPT_FOLLOWER_ATOMIC":
        return GPTFollowerAtomic(config_data.config)
    elif config_data.agent_type == "GPT_FOLLOWER":
        return GptFollower(config_data.config)
    elif config_data.agent_type == "OLLAMA_MODELS":
        return Models(config_data.config)
    else:
        raise ValueError(f"Unknown agent type: {config_data.agent_type}")