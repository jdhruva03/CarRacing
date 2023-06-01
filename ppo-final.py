import gymnasium as gym

from stable_baselines3 import ppo
from stable_baselines3 import PPO


from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import os


# Declare the environment
env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False, render_mode="human")


ppo_model_final = "Models//PPO//final"
log_dir = "Logs"

if not os.path.exists(ppo_model_final):
    os.makedirs(ppo_model_final)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

ppo_timeSteps = 1000000

ppo_model= PPO("CnnPolicy", env, verbose=1, learning_rate=0.0005, gamma=0.99, batch_size=256, clip_range=0.2,  device="cuda", tensorboard_log=log_dir) 
ppo_log_name = f"ppo_final_steps_{ppo_timeSteps}"
ppo_model.learn(total_timesteps=ppo_timeSteps, log_interval=1, tb_log_name=ppo_log_name)

ppo_path = f"{ppo_model_final}//ppo_final_steps{ppo_timeSteps}"
ppo_model.save(ppo_path)
env.reset()