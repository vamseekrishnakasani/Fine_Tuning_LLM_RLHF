# Fine-Tuning AI Using RLHF (Reinforcement Learning from Human Feedback)

This project demonstrates how to fine-tune a language model using RLHF, leveraging custom reward modeling and preference datasets. The workflow includes building a reward model, training it on human preference data, and then using it to optimize a language model with Proximal Policy Optimization (PPO).

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Workflow](#workflow)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Prompt Dataset Creation (IMDB)](#2-prompt-dataset-creation-imdb)
  - [3. Preference Dataset Creation (TULU)](#3-preference-dataset-creation-tulu)
  - [4. Custom Reward Model](#4-custom-reward-model)
  - [5. Reward Model Training](#5-reward-model-training)
  - [6. PPO Fine-Tuning](#6-ppo-fine-tuning)
  - [7. Evaluation](#7-evaluation)
  - [8. Saving Models](#8-saving-models)
- [Usage](#usage)
- [Notes](#notes)
- [References](#references)

---

## Overview

This script walks through the process of fine-tuning a GPT-2 language model using RLHF. It includes:
- Building a prompt dataset from IMDB reviews.
- Creating a preference dataset from the TULU dataset.
- Training a custom reward model to distinguish between preferred and non-preferred responses.
- Using PPO to optimize the language model with the reward signal.
- Evaluating the improvement in model responses.

## Requirements

- Python 3.8+
- CUDA-enabled GPU (recommended for training)
- The following Python packages:
  - numpy < 2
  - torch == 2.6.0
  - transformers == 4.39.1
  - trl == 0.8.6
  - datasets == 2.18.0
  - peft == 0.10.0
  - tqdm == 4.66.2
  - pandas

You can install the required packages using pip:

```bash
pip install 'numpy<2' torch==2.6.0 transformers==4.39.1 trl==0.8.6 datasets==2.18.0 peft==0.10.0 tqdm==4.66.2 pandas
```

## Workflow

### 1. Environment Setup
- Installs all required packages.
- Detects and prints the device (CPU or GPU).

### 2. Prompt Dataset Creation (IMDB)
- Loads the IMDB dataset using Hugging Face Datasets.
- Filters reviews longer than 200 tokens.
- Tokenizes and truncates reviews to create prompts for language model training.

### 3. Preference Dataset Creation (TULU)
- Loads the `anthropic_hh` split from the TULU preference dataset.
- Extracts user prompts and assistant responses, forming pairs of (chosen, rejected) responses.
- Converts the preference dataset into a comparison dataset for reward modeling.

### 4. Custom Reward Model
- Defines a reward model based on DistilGPT2 with a linear head.
- The model scores (prompt, response) pairs.

### 5. Reward Model Training
- Trains the reward model using contrastive loss: the model learns to score preferred responses higher than rejected ones.
- Saves the trained model and tokenizer for later use.

### 6. PPO Fine-Tuning
- Loads a GPT-2 model ("lvwerra/gpt2-imdb") with a value head for PPO.
- Uses the custom reward model or a sentiment classifier (DistilBERT) as the reward function.
- Runs PPO training, optimizing the language model to generate more preferred responses.

### 7. Evaluation
- Compares the base and PPO-tuned models on a batch of prompts.
- Generates responses from both models and scores them using the reward model or sentiment classifier.
- Outputs summary statistics (mean/median rewards before and after PPO).

### 8. Saving Models
- Saves the PPO-tuned model and tokenizer for future inference.

## Usage

1. **Clone the repository and navigate to the project directory.**
2. **Install the required packages** (see [Requirements](#requirements)).
3. **Run the script:**
   - The script is designed to be run as a standalone Python file:
   ```bash
   python fine_tuning_ai_using_rlhf.py
   ```
   - (If running in a notebook, remove or comment out the `os.kill(os.getpid(), 9)` line and the `!pip install ...` lines.)
4. **Monitor the output:**
   - The script prints progress for each major step, including training loss, evaluation results, and summary statistics.
5. **Check saved models:**
   - The trained reward model is saved as `reward_model.pt` and tokenizer in `reward_tokenizer/`.
   - The PPO-tuned language model is saved in `gpt2-imdb-pos-v2/`.

## Notes
- The script is resource-intensive and may require a GPU for reasonable training times.
- The reward model is trained on a subset (5,000 samples) of the TULU dataset for demonstration.
- PPO training uses a sentiment classifier as a reward function for simplicity, but you can swap in the custom reward model.
- The script is modular and can be adapted for other datasets or reward models.

## References
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- [IMDB Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
- [TULU Preference Dataset](https://huggingface.co/datasets/allenai/preference-datasets-tulu)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)

---