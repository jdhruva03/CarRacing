import gymnasium as gym

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False, render_mode="human")

# model = PPO.load("C://CarRacing//Models//ppo_model_500000")
model = DQN.load("C://CarRacing//Models//dqn_model_500000")

print("Evaluating")
evaluate_policy(model, env, n_eval_episodes=1, render=True)
env.close()

'''
obs = env.reset()
while True:
    action = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
'''