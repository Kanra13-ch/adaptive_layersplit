import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
import torch
import torch.nn as nn

class RewarfPredictor(nn.Module):
    def __init__(self):
        super(RewarfPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)

reward_predictor_path = 'reward_predictor.pth'
reward_predictor = RewarfPredictor()
reward_predictor.load_state_dict(torch.load(reward_predictor_path))
reward_predictor.eval()

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # self.action_space = spaces.Discrete(5)
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([0.3, 11]),
                                            dtype=np.float32)

        self.state = None
        self.max_steps_per_episode = 5
        self.reset()


    def reset(self):
        self.current_step = 0
        noise_level = np.random.uniform(low=0.01, high=0.3)
        split_point = np.random.randint(low=1, high=10)
        self.state = np.array([noise_level, split_point], dtype=np.float32)
        return self.state

    def calculate_reward(self, state, action):
        state_tensor = torch.tensor([state], dtype=torch.float32)  # 转换为2D Tensor
        with torch.no_grad():
            inference_reward = reward_predictor(state_tensor).item()

        split_point = state[1]
        max_split_point = 10
        load_reward = (max_split_point - split_point) / max_split_point
        inference_factor = 10
        load_factor = 1
        total_reward = inference_factor * inference_reward + load_factor * load_reward
        return total_reward, inference_reward, load_reward

    def step(self, action):
        self.current_step += 1
        noise_level, split_point = self.state

        if split_point == 1 and action in [1, 3, 5, 7]:
        # if split_point == 1 and action == 1:
            action = 0
        elif split_point == 10 and action in [2, 4, 6, 8]:
        # elif split_point == 10 and action == 2:
            action = 0
        if split_point == 2 and action in [3, 5, 7]:
        # if split_point == 1 and action == 1:
            action = 1
        elif split_point == 9 and action in [4, 6, 8]:
        # elif split_point == 10 and action == 2:
            action = 2
        if split_point == 3 and action in [5, 7]:
        # if split_point == 1 and action == 1:
            action = 3
        elif split_point == 8 and action in [6, 8]:
        # elif split_point == 10 and action == 2:
            action = 4
        if split_point == 4 and action in [7]:
        # if split_point == 1 and action == 1:
            action = 5
        elif split_point == 7 and action in [8]:
        # elif split_point == 10 and action == 2:
            action = 6
        noise_adjustment_factor = 1.0

        if action == 1:  # 向下移1层
            split_point = max(1, split_point - 1)
            noise_adjustment_factor = 1.0005
        elif action == 2:  # 向上移1层
            split_point = min(10, split_point + 1)
            noise_adjustment_factor = 0.9995
        elif action == 3:  # 向下移2层
            split_point = max(1, split_point - 2)
            noise_adjustment_factor = 1.001
        elif action == 4:  # 向上移2层
            split_point = min(10, split_point + 2)
            noise_adjustment_factor = 0.999
        elif action == 5:  # 向下移2层
            split_point = max(1, split_point - 3)
            noise_adjustment_factor = 1.0015
        elif action == 6:  # 向上移2层
            split_point = min(10, split_point + 3)
            noise_adjustment_factor = 0.9995
        elif action == 7:  # 向下移2层
            split_point = max(1, split_point - 4)
            noise_adjustment_factor = 1.002
        elif action == 8:  # 向上移2层
            split_point = min(10, split_point + 4)
            noise_adjustment_factor = 0.998


        noise_level *= noise_adjustment_factor
        noise_level = min(max(noise_level, 0), 0.3)

        state = np.array([noise_level, split_point], dtype=np.float32)
        reward, inference_reward, load_reward = self.calculate_reward(state, action)

        self.state = state
        done = self.current_step >= self.max_steps_per_episode

        return self.state, reward, done, {}

env = CustomEnv()
tensorboard_log_path = "./neo/tensorboard_logs/"

ppo_model = PPO("MlpPolicy", env, n_steps=400, verbose=1, batch_size=100, tensorboard_log=tensorboard_log_path)

ppo_model.learn(total_timesteps=240000, tb_log_name="act_9")
ppo_model.save("./neo/act_9")