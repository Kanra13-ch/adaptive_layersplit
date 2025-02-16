import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from subclassTheLlama import LlamaForCausalLMWithChannel, calculate_ppl
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

model_id = "/data/cyx_res/results/original_pretrained_model"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = LlamaForCausalLMWithChannel.from_pretrained(model_id,
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                             )
device = "cuda"
test = load_dataset('parquet', data_files='/home/zu/cyx/cyx_llm/test.parquet', split="train")

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=np.array([0, 1]), high=np.array([0.3, 10]),
                                            dtype=np.float32)

        self.state = None
        self.PPL_old = 0
        self.max_steps_per_episode = 5
        #self.stable_steps_threshold = 3
        self.reset()


    def reset(self):
        self.current_step = 0
        self.stable_steps_count = 0
        noise_level = np.random.uniform(low=0.08, high=0.22)
        split_point = np.random.randint(low=3, high=7)
        model.set_dropout_prob(noise_level)
        model.set_specified_layers(split_point)
        self.PPL_initial = calculate_ppl(model, tokenizer, test, device, num_runs=1)
        self.state = np.array([noise_level, split_point], dtype=np.float32)
        return self.state

    def calculate_reward(self, PPL_initial, PPL_new, split_point):
        inference_factor = 10.5
        load_factor = 1
        max_split_point = 10
        inference_reward = (PPL_initial - PPL_new) / PPL_initial if PPL_initial != 0 else 0
        load_reward = (max_split_point - split_point) / max_split_point

        total_reward = inference_factor * inference_reward + load_factor * load_reward
        return total_reward, inference_reward, load_reward

    def step(self, action):
        self.current_step += 1
        noise_level, split_point = self.state

        new_split_point = action + 1
        if new_split_point > split_point:
            noise_adjustment_factor = 0.9995
        elif new_split_point < split_point:
            noise_adjustment_factor = 1.0005
        else:
            noise_adjustment_factor = 1

        noise_level *= noise_adjustment_factor
        noise_level = min(max(noise_level, 0), 0.3)

        model.set_dropout_prob(noise_level)
        model.set_specified_layers(new_split_point)
        if new_split_point == split_point:
            PPL_new = self.PPL_old
        else:
            PPL_new = calculate_ppl(model, tokenizer, test, device, num_runs=1)
            diff_ppl = abs(PPL_new - self.PPL_old)


        reward, inference_reward, load_reward = self.calculate_reward(self.PPL_initial, PPL_new, new_split_point)
        self.PPL_old = PPL_new

        self.state = np.array([noise_level, new_split_point], dtype=np.float32)
        done = self.current_step >= self.max_steps_per_episode

        with open("/home/zu/cyx/cyx_llm/log_new_fac105.txt", "a") as log_file:
            log_file.write(f"Step: {self.current_step}, Action: {action}, New State: {self.state}, Reward: {reward},"
                           f"i_r: {inference_reward},  l_r: {load_reward}\n")

        return self.state, reward, done, {}

env = CustomEnv()

tensorboard_log_path = "./tensorboard_logs/"

ppo_model = PPO("MlpPolicy", env, n_steps=100, verbose=2, batch_size=25, tensorboard_log=tensorboard_log_path)

ppo_model.learn(total_timesteps=6000, tb_log_name="ppo_run")
ppo_model.save("/home/zu/cyx/cyx_llm/ppo_log_new_fac105")