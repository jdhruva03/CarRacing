import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3 import PPO


from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import os


# Declare the environment
env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False, render_mode="human")

dqn_models_final = "Models//DQN//final"
log_dir = "Logs"

if not os.path.exists(dqn_models_final):
    os.makedirs(dqn_models_final)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

DQN_timeSteps = 1000000
BufferSize = 100000
LearningStarts = 50000


dqn_model= DQN("CnnPolicy", env, verbose=1, learning_rate=0.0001, gamma=0.99, batch_size=64, train_freq=8, buffer_size=BufferSize, learning_starts=LearningStarts,  device="cuda", tensorboard_log=log_dir) 
dqn_log_name = f"dqn_final_steps_{DQN_timeSteps}"
dqn_model.learn(total_timesteps=DQN_timeSteps, log_interval=1, tb_log_name=dqn_log_name)

dqn_path = f"{dqn_models_final}//dqn_final_steps{DQN_timeSteps}"
dqn_model.save(dqn_path)
env.reset()