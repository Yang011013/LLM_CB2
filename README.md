<a id="readme-top"></a>

<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
# Synergizing LLMS: From Closed-Source Prototyping to Open-Source Model Enhancement in Instruction Following

**This Repository is modified based on [cb2](https://github.com/lil-lab/cb2).**

<video width="640" height="360" controls>
  <source src="ab473981fb151765736fcb48dfdbaa30_raw.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

# Abstract

We study the problem of constructing an efficient LLM-based instruction-following agent capable of comprehending and executing open-ended instructions in an embodied environment. We propose a method called Synergizing LLMs for rapid domain adaptation in the instruction-following task without requiring additional manual annotation. This approach leverages a large general-purpose LLM to establish task baselines and generate domain-specific data. The knowledge from the larger model is then gradually transferred to a domain-tuned open-source LLM through a model transition process, enabling faster and more efficient adaptation. Accordingly, we developed the Dynamic Instruction Decomposition (DID) framework, specifically designed for LLM integration within this task scenario. The DID framework enables the agent to progressively align open-ended natural language commands with dynamic environmental contexts. Experimental results demon strate significant improvements in task accuracy, leading to more effective instruction following and enhanced human-agent collaboration.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

# Getting Started

## Installation

- CB2_LLM requires `Python 3.9` or higher.
- You must have an NVIDIA graphics card with at least 24GB VRAM and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed if you want to use the fintued LLM as follower in the game.

```sh
# Recommand to use annoconda
conda create -n llm_cb2 python=3.9
conda activate llm_cb2
# Clone the repo
git clone https://github.com/Yang011013/LLM_CB2.git
cd llm_cb2

# Build the environment
## Install packages
pip install -r requirement.txt
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

To play the CB2 game with an LLM, follow these steps:

1. **Start the server:**

   ```sh
   python -m cb2game.server.main --config_filepath server_config/llm_server_config.yaml
   ```

   Games between you and the LLM will be saved in `game_database/human_llm/`.

2. **Choose an LLM as your follower:**

   - **Gemini or GPT:**
     - Get your Gemini API key from [here](https://aistudio.google.com/apikey) and add it to:
       `src/cb2game/agents/agent_config_files/gemini_follower_atomic.yaml`.
     - Get your OpenAI API key from [here](https://platform.openai.com/api-keys) and add it to:
       `src/cb2game/agents/agent_config_files/gpt_follower_atomic.yaml`.
   
     To run with Gemini or GPT:

     ```sh
     python -m cb2game.agents.remote_vision_agent --host "http://localhost:8080" --agent_config_filepath <path_to_yaml_file>
     ```

   - **Finetuned Mixtral:**
     - Download the model from Hugging Face: [mistral-7b](https://huggingface.co/unsloth/mistral-7b-v0.3-bnb-4bit) to `models/mistral-7b-v0.3-bnb-4bit`.
     - Get the finetuned weights [here]().

     To run with Mixtral:

     ```sh
     python -m cb2game.agents.remote_vision_agent --host "http://localhost:8080" --agent_config_filepath src/cb2game/agents/agent_config_files/fintuned_mixtral.yaml
     ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Acknowledgement -->

# Acknowledgement

Our code is based on [cb2](https://github.com/lil-lab/cb2). We thank the authors for their effort in building such a great codebase.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

# Citation



<p align="right">(<a href="#readme-top">back to top</a>)</p>
