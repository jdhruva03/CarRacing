import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import os



#---
env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False, render_mode="human")

print("Training")
log_path = "C://CarRacing//Logs//"
model = PPO("CnnPolicy", env, verbose=1, device="cuda")
model.learn(total_timesteps=500000)

ppo_path = "C://CarRacing//Models//ppo_model_500000"
model.save(ppo_path)

print("Evaluating")
evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()



'''

# Parallel environments
env = make_vec_env("CartPole-v1", n_envs=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()



'''