# Adaptive Layer Splitting for Wireless LLM Inference

This repository demonstrates the use of reinforcement learning (RL) for optimizing large language model (LLM) inference across edge networks. The project leverages Proximal Policy Optimization (PPO) and a reward surrogate model (RSM) to optimize the split layers of Llama-based models under varying network conditions.

## Files Overview

1. **train.py**: This file contains the PPO-based reinforcement learning training script. It defines a custom environment where the agent optimizes the noise level and split points of the model. The reward surrogate model (RSM) is used to calculate the rewards for each action taken.

2. **train_rsm.py**: Similar to `train.py`, this script trains the model using PPO with a reward surrogate model integrated into the environment. It further optimizes the dropout probability and model layer splitting for performance enhancement.

3. **add_noise.py**: This script contains a subclass of the Llama model, which adds noise and simulates delays in the network. It provides a method for adjusting the dropout probability and simulating real-world noise conditions during the inference process.

## Setup Instructions

### Requirements

- Python 3.8+
- PyTorch
- Transformers
- Stable Baselines3
- Datasets (for evaluation)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### How to Run

1. **Training PPO Agent**:
    To train the PPO agent with the custom environment:

   ```bash
   python train.py
   ```

2. **Training with Reward Surrogate Model**:
    To train the PPO agent using the reward surrogate model:

   ```bash
   python train_rsm.py
   ```

3. **Model Evaluation and Testing with Noise**:
    To simulate network noise and adjust model layers during inference:

   ```bash
   python add_noise.py
   ```

