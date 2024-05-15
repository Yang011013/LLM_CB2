""" Defines config for configuring CB2 agents. """
import dataclasses
import importlib
import inspect
import logging
from enum import Enum
from typing import Optional, Union

import yaml
from mashumaro.mixins.json import DataClassJSONMixin

from cb2game.agents.agent import Agent
from cb2game.agents.gpt_follower import GptFollower, GptFollowerConfig
from cb2game.agents.simple_follower import SimpleFollower, SimpleFollowerConfig
from cb2game.agents.gemini_follower import GeminiFollower, GeminiFollowerConfig
from cb2game.agents.gemini_follower_atomic import GeminiFollowerAtomic, GeminiFollowerConfigAtomic
from cb2game.agents.gemini_vision_follower import GeminiVisionFollower, GeminiVisionFollowerConfig
from cb2game.agents.gemini_vision_follower_atomic import GeminiVisionFollowerAtomic, GeminiVisionFollowerConfigAtomic
from cb2game.agents.claude_follower import ClaudeFollower, ClaudeFollowerConfig
from cb2game.agents.claude_vision_follower import ClaudeVisionFollower, ClaudeVisionFollowerConfig
from cb2game.agents.baseline_follower import BaselineFollower, BaselineFollowerConfig
from cb2game.util.deprecated import deprecated

logger = logging.getLogger(__name__)


@deprecated("Use AgentConfigData and LoadAgentFromConfig() instead.")
class AgentType(Enum):
    NONE = 0
    # Follower used for CB2 pilot study.
    PILOT_FOLLOWER = 1
    # Experimental follower that uses a text-only interface to OpenAI's GPT API.
    GPT_FOLLOWER = 2
    # Simple follower/leader for unit testing and debugging.
    SIMPLE_FOLLOWER = 3
    SIMPLE_LEADER = 4

    def __str__(self):
        return self.name

    @staticmethod
    def from_str(s: str):
        return AgentType[s]


# @deprecated("Use AgentConfigData and LoadAgentFromConfig() instead.")
@dataclasses.dataclass
class AgentConfig(DataClassJSONMixin):
    name: str
    comment: str
    # agent_type must be one of the values in enum AgentType.
    agent_type: str
    config: Union[GptFollowerConfig,
        SimpleFollowerConfig,
        GeminiFollowerConfig,
        GeminiFollowerConfigAtomic,
        GeminiVisionFollowerConfig,
        GeminiVisionFollowerConfigAtomic,
        ClaudeFollowerConfig,
        ClaudeVisionFollowerConfig,
        BaselineFollowerConfig,
    ]


@dataclasses.dataclass
class AgentConfigData:
    name: str  # Name of the agent.
    comment: str  # Comment for the agent.
    type: str  # Agent type. Contains module and class names.
    config: dict  # Configuration for initializing the agent.

def ReadAgentConfigOrDie(config_path) -> AgentConfig:
    with open(config_path, "r") as cfg_file:
        data = yaml.load(cfg_file, Loader=yaml.SafeLoader)
    # If the resulting type is not AgentConfigData, then convert it to one, by
    # initializing the AgentConfigData object with the dictionary.
    if not isinstance(data, AgentConfig):
        try:
            data = AgentConfig(**data)
        except TypeError as e:
            logger.error("Error parsing agent config: %s", e)
            raise
    return data


def SerializeAgentConfig(config: AgentConfigData) -> str:
    return yaml.dump(config)

def LoadAgentFromConfig(config_data: AgentConfig) -> Agent:
    if config_data.agent_type == "SIMPLE_FOLLOWER":
        return SimpleFollower(config_data.config)
    elif config_data.agent_type == "GPT_FOLLOWER":
        return GptFollower(config_data.config)
    elif config_data.agent_type == "GEMINI_FOLLOWER":
        return GeminiFollower(config_data.config)
    elif config_data.agent_type == "GEMINI_FOLLOWER_ATOMIC":
        return GeminiFollowerAtomic(config_data.config)
    elif config_data.agent_type == "GEMINI_VISION_FOLLOWER":
        return GeminiVisionFollower(config_data.config)
    elif config_data.agent_type == "GEMINI_VISION_FOLLOWER_ATOMIC":
        return GeminiVisionFollowerAtomic(config_data.config)
    elif config_data.agent_type == "CLAUDE_FOLLOWER":
        return ClaudeFollower(config_data.config)
    elif config_data.agent_type == "CLAUDE_VISION_FOLLOWER":
        return ClaudeVisionFollower(config_data.config)
    elif config_data.agent_type == "BASELINE_FOLLOWER":
        return BaselineFollower(config_data.config)
    else:
        raise ValueError(f"Unknown agent type: {config_data.agent_type}")