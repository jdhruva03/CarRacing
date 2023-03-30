import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False, render_mode="human")

model = DQN("CnnPolicy", env, verbose=1,  buffer_size=20000, device="cuda")
model.learn(total_timesteps=500000)

dqn_path = "C://CarRacing//Models//dqn_model_500000"
model.save(dqn_path)

del model # remove to demonstrate saving and loading

model = DQN.load("C://CarRacing//Models//dqn_model_500000")



print("Evaluating")
evaluate_policy(model, env, n_eval_episodes=10, render=True)
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