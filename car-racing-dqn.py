'''
Author: Jainam Dhruva
Title: Deep Q learning on autonomous car agent
'''


# python C:\Users\Jainam\anaconda3\envs\ml\Lib\site-packages\gymnasium\envs\box2d\car_racing.py

import gymnasium as gym
import torch 
import torch.nn
from collections import deque
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import cv2
from skimage import color, transform, io



class processimage:
    def process_image(obs):
        #uncomment to see original image
        # plt.imshow(obs)
        # plt.show()

        obs1 = obs.astype(np.uint8)
        obs_gray = color.rgb2gray(obs1)
        # obs_gray[abs(obs_gray - 0.60116) < 0.1] = 1
        obs_gray[84:95,0:12] = 0
        obs_gray[abs(obs_gray - 0.68616) < 0.0001] = 1
        obs_gray[abs(obs_gray - 0.75630) < 0.0001] = 1
        
        #uncomment to see pre processed image
        # plt.imshow(obs_gray, cmap='gray')
        # plt.show()

        #Set values between -1 and 1 for input normalization
        return 2 * obs_gray - 1


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# HYPER PARAMETERS
DECAY = 0.01
MIN_EPSILON = 0.05
MAX_EPSILON = 1
TOTAL_EPISODESS = 300
MAX_EXPERIENCES = 20000
TRAIN_TARGET_MODEL_STEPS = 100
LEARNING_RATE = 0.05
GAMMA = 0.618
BATCH_SIZE = 64
KEEP_PROB = 0.8


# Implementation of CNN/ConvNet Model
class DQN(torch.nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=8, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - KEEP_PROB)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - KEEP_PROB)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=1 - KEEP_PROB)
        )

        # L4 FC 506944 121 7744 inputs -> 64 outputs
        self.fc1 = torch.nn.Linear(7744, 64, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - KEEP_PROB)
        )
        # L5 Final FC 64 inputs -> 5 outputs
        self.fc2 = torch.nn.Linear(64, 5, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters

    def forward(self, x):
        # print(f"A Dimension of input: {x.shape}")
        out = self.layer1(x)
        # print(f"B Dimension of input after layer 1 (Conv2d): {out.shape}")
        out = self.layer2(out)
        # print(f"C Dimension of input after layer 2 (Conv2d): {out.shape}")
        out = self.layer3(out)
        # print(f"D Dimension of input after layer 3 (Conv2d): {out.shape}")
        # print(out)
        # print("out [0] ", out[0], " out.size(0) ", out.size(0) )
        # input("yo")
        out = out.view(out.size(0), -1)   # Flatten them for FC
        # print(f"E Dimension of input after Flattening (Conv2d): {out.shape}")
        # input("here2")
        out = self.fc1(out)
        # print(f"F Dimension of input after layer 4 (forward1): {out.shape}")
        # input("here3")
        out = self.fc2(out)
        # print(f"G Dimension of input after layer 5 (forward2): {out.shape}")
        # input("here4, by return")
        # print(out)
        return out

    def act(self, current_state):
        ## Convert the state to a tensor
        current_state_tensor = torch.as_tensor(current_state, dtype=torch.float32)
        ## Shape the tensor for input into the network
        current_state_tensor = (current_state_tensor.unsqueeze(0)).unsqueeze(0)
        ## Forward pass
        current_state_tensor = current_state_tensor.cuda()
        Q_values = self.forward(current_state_tensor)
        ## Get Max
        max_Q_val_index = action = torch.argmax(Q_values, dim=1)[0]

        ## Detach - Important
        action = max_Q_val_index.detach().item()

        return action

        
def train(env, replay_buffer, prediction_model, target_model, episode_done, episode, CRITERION, OPTIMIZER):
    # print("Training: ")
    torch.autograd.set_detect_anomaly(True)

    ## Setting the model to training model
    prediction_model.train()   

    ## HYPER PARAMETERS
    global LEARNING_RATE 
    global GAMMA 

    MIN_REPLAY_SIZE = 100
    ## Only train if atleast minimum replay size has been met
    if len(replay_buffer) < MIN_REPLAY_SIZE:
        return

    ## Create a batch for training
    global BATCH_SIZE
    mini_batch = random.sample(replay_buffer, BATCH_SIZE)
    
    current_states = np.asarray([transition[0] for transition in mini_batch])
    actions = np.asarray([transition[1] for transition in mini_batch])
    rewards = np.asarray([transition[2] for transition in mini_batch])
    new_states = np.asarray([transition[3] for transition in mini_batch])
    episode_dones = np.asarray([transition[4] for transition in mini_batch])

    current_states_tensor = torch.as_tensor(current_states, dtype=torch.float32)
    actions_tensor = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
    new_states_tensor = torch.as_tensor(new_states, dtype=torch.float32)
    episode_dones_tensor = torch.as_tensor(episode_dones, dtype=torch.float32).unsqueeze(-1)

    current_states_tensor = current_states_tensor.cuda()
    actions_tensor = actions_tensor.cuda()
    rewards_tensor = rewards_tensor.cuda()
    new_states_tensor = new_states_tensor.cuda()
    episode_dones_tensor = episode_dones_tensor.cuda()

    ## Compute Targets
    new_states_tensor = (new_states_tensor.squeeze(0)).unsqueeze(1)
    new_states_tensor = new_states_tensor.cuda()
    target_Q_values = target_model(new_states_tensor)
    max_Target_Q_values = target_Q_values.max(dim=1, keepdim=True)[0]

    Targets = (rewards_tensor) + (GAMMA * (1-episode_dones_tensor) * max_Target_Q_values)

    # HYPER PARAMETERS
    # CRITERION = torch.nn.MSELoss()
    # OPTIMIZER = torch.optim.Adam(prediction_model.parameters(), lr=LEARNING_RATE)

    ## Compute Loss and Update Gradients
    current_states_tensor = (current_states_tensor.squeeze(0)).unsqueeze(1)
    current_states_tensor = current_states_tensor.cuda()
    Q_values = prediction_model(current_states_tensor)
    action_Q_values = torch.gather(input=Q_values, dim=1, index=actions_tensor)

    action_Q_values, Targets = action_Q_values.cuda(), Targets.cuda()
    loss = CRITERION(action_Q_values, Targets)
    
    OPTIMIZER.zero_grad
    loss.backward()
    OPTIMIZER.step()
    
    return loss
   

    
def main():

    ## Intitialize the Environment
    env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False, render_mode="human")

    ## Initialize the Replay Buffer
    replay_buffer = deque(maxlen=MAX_EXPERIENCES)
    reward_buffer = deque([0,0], maxlen=100)
    
    ## Initialize the Target and Main models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## Main Model 
    prediction_model = DQN()
    prediction_model = prediction_model.cuda()
    ## Target Model 
    target_model = DQN()
    target_model = target_model.cuda()
    ## Set target weights = prediction weights 
    target_model.load_state_dict(prediction_model.state_dict())

    # HYPER PARAMETERS
    CRITERION = torch.nn.MSELoss()
    OPTIMIZER = torch.optim.Adam(prediction_model.parameters(), lr=LEARNING_RATE)

    LOSS = 0

    print(prediction_model)
    print(target_model)

    

    # Step Count to keep track of when to update prediction_model and target_model
    step_count = 0
    
    for episode in range(TOTAL_EPISODESS):

        episode_reward = 0
        
        current_state, info = env.reset()
        current_state = processimage.process_image(current_state)
        
        episode_done = False
        epsilon = 0.75

        print("EPISODE " + str(episode))

        while (not episode_done):

            step_count += 1
            # print(step_count)
            if(step_count%500==0):
                print("Step Count = " + str(step_count))

            # Use Q value to determine action
            random_number = np.random.random()
            if(random_number < epsilon):
                # Do random action
                action = env.action_space.sample()
                # print("Acttion random", action, type(action))  
                # TBD - assign more randomeness to accelerate so the car moves ahead

            else:
                action = prediction_model.act(current_state)

                 
                

            # 'current_  state/new_state' is an 96*96*3 array containing RGB values for the 96*96 pixel image
            new_state, reward, terminated, truncated, info = env.step(action)
            new_state = processimage.process_image(new_state)
            
            if (terminated or truncated):
                print("Episode done omg")
                episode_done = True
            
            # Add the experience to the replay Buffer
            transition_buffer = (current_state, action, reward, new_state, episode_done)    
            replay_buffer.append(transition_buffer)

            #  Update the Main Network using the Bellman Equation
            if ( (step_count % 4 == 0) or episode_done):
                LOSS = train(env, replay_buffer, prediction_model, target_model, episode_done, episode, CRITERION, OPTIMIZER)
                # dummy_function(step_count)

            ## Set the new state as current state
            current_state = new_state
            episode_reward += reward


            # If step count is greater than equal to train_model_steps
            if (step_count >= TRAIN_TARGET_MODEL_STEPS):
                # print('Copying main network weights to the target network weights')
                target_model.load_state_dict(prediction_model.state_dict())
                step_count = 0
                

            # In case the episode is done, handle that
            if (episode_done):
                print('Total training rewards: {} after n episodes = {} with final reward = {}'.format(episode_reward, episode, reward))
                print('Loss: ', LOSS)
                episode_reward += 1
                break
        
        ## Add episode reward to the reward buffer
        reward_buffer.append(episode_reward)

        # Update epsilon after each episode
        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY * episode)

        # Save models every 5 episodes
        if (episode%5==0):
            PATH = "C://CarRacing//Models//prediction_model.pt"
            torch.save({
            'episode': episode,
            'model_state_dict': prediction_model.state_dict(),
            'optimizer_state_dict': OPTIMIZER.state_dict(),
            'loss': LOSS,
            'epsilon': epsilon,
            'replay_buffer': replay_buffer,
            }, PATH)

            PATH2 = "C://CarRacing//Models//target_model.pt"
            torch.save({
            'episode': episode,
            'model_state_dict': target_model.state_dict(),
            'optimizer_state_dict': OPTIMIZER.state_dict(),
            'loss': LOSS,
            'epsilon': epsilon,
            'replay_buffer': replay_buffer,
            }, PATH2)
    
    
    env.close()

if __name__ == '__main__':
    main()