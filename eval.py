import gymnasium as gym

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False, render_mode="human")

model = PPO.load("C://CarRacing//Models//ppo//final//ppo_final_steps1000000")
print(model.policy)
modeldqn = DQN.load("C://CarRacing//Models//dqn_model_500000")
print("dqn")
print(modeldqn.policy)

print("Evaluating")
evaluate_policy(model, env, n_eval_episodes=3, render=True)
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