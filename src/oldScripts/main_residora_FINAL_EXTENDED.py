import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F
from typing import List

import re
import numpy as np
import requests
import json
from pprint import pprint
import os
from datetime import datetime
from copy import deepcopy

from typing import List

import pandas as pd
import nltk
import matplotlib.pyplot as plt

from src.mainHelpers.prepare4APIRequest import prepare4APIRequest
from src.residoraHelpers.convNet import convNet
from src.residoraHelpers.ActorCritic import ActorCritic
from src.residoraHelpers.PPO import PPO

#---------------------------------------------------------------------------------

#setup env

####### initialize environment hyperparameters ######

env_name = "CartPole-v1"
env_name = "Acrobot-v1"
has_continuous_action_space = False

max_ep_len = 400                    # max timesteps in one episode
max_training_timesteps = int(1e5)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)
save_model_freq = int(2e4)      # save model frequency (in num timesteps)

action_std = None
update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 40               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)
env = gym.make(env_name)

# state space dimension
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

log_dir = "PPO_logs"
if not os.path.exists(log_dir):
      os.makedirs(log_dir)

log_dir = log_dir + '/' + env_name + '/'
if not os.path.exists(log_dir):
      os.makedirs(log_dir)

################### checkpointing ###################

run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

directory = "models/residora/"
checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)

################### Let go ###################
torch.cuda.empty_cache()
ppo_agent = None

actorCritic = ActorCritic(
    state_dim=state_dim,
    action_dim=action_dim,
    lr_actor=lr_actor,
    lr_critic=lr_critic,
    nrNeuronsInHiddenLayers=[64,64],
    activationFunction="tanh",
    seed = 13,
    useCNN = True,
    stride = 1,
    kernel_size=3,
    out_channel=4,
    dropOutProb=0.01
)

""" actorCritic = ActorCritic(
    state_dim=state_dim,
    action_dim=action_dim,
    lr_actor=lr_actor,
    lr_critic=lr_critic,
    nrNeuronsInHiddenLayers=[64,64],
    activationFunction="tanh",
    seed = 13,
    useCNN = False,
    dropOutProb=0.01
) """

ppo_agent = PPO(
    ActorCritic=actorCritic,
    gamma=gamma,
    K_epochs=K_epochs,
    eps_clip=eps_clip,
    device="cpu"
)


#----------------
def run():
    start_time = datetime.now().replace(microsecond=0)
    print_running_reward = 0
    print_running_episodes = 0
    log_running_reward = 0
    log_running_episodes = 0
    time_step = 0
    i_episode = 0

    #logfile
    log_f = open(log_dir + "/test.csv","w+")
    log_f.write('episode,timestep,reward\n')

    while time_step <= max_training_timesteps:
        
        state, _ = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):
            
            # select action with policy
            action = ppo_agent.select_action_exploitation(state)
            state, reward, done, _, dic_ = env.step(action)
            #print(state, reward, done, _, dic_)
            # saving reward and is_terminals
            ppo_agent.rewards.append(reward)
            ppo_agent.isTerminals.append(done)
            
            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

                
            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")
                
            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        
        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
#####################################################

run()

#####################################################
#------ Ploting ----------

plotTraining(path = "/home/cewinharhar/GITHUB/reincatalyze/PPO_logs/Acrobot-v1/test.csv" )
plotTraining(path = "/home/cewinharhar/GITHUB/reincatalyze/PPO_logs/CartPole-v1/test.csv" )

def plotTraining(path = "/home/cewinharhar/GITHUB/reincatalyze/PPO_logs/CartPole-v1/test.csv"):
    data = pd.read_csv(path)
    # smooth out rewards to get a smooth and a less smooth (var) plot lines
    fig_width = 10
    fig_height = 6


    # smooth out rewards to get a smooth and a less smooth (var) plot lines
    window_len_smooth = 10
    min_window_len_smooth = 1
    linewidth_smooth = 1.5
    alpha_smooth = 1

    window_len_var = 5
    min_window_len_var = 1
    linewidth_var = 2
    alpha_var = 0.1
    ax = plt.gca()
    data_avg = data.copy()

    data_avg['reward_smooth'] = data['reward'].rolling(window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
    data_avg['reward_var'] = data['reward'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()

    data_avg.plot(kind='line', x='timestep' , y='reward_smooth',ax=ax,color="red",  linewidth=linewidth_smooth, alpha=alpha_smooth)
    data_avg.plot(kind='line', x='timestep' , y='reward_var',ax=ax,color="blue",  linewidth=linewidth_var, alpha=alpha_var)

    # keep only reward_smooth in the legend and rename it
    handles, labels = ax.get_legend_handles_labels()

    plt.show()

""" env.close() """


""" aKGD31 = "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"

input_size = 1024
hidden_size = [512, 256]
output_size = len(list(aKGD31)) """
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################



def reward_function(originalSeq_, new_stateSeq_):
    # Calculate reward based on state and action
    # Here's a simple example:
    if isinstance(originalSeq_, list):
        originalSeq_ = "".join(originalSeq_)
    if isinstance(new_stateSeq_, list):
        new_stateSeq_ = "".join(new_stateSeq_)

    #TODO remove
    originalSeq_ = "MSTETLRLQKARATEEGLAFETPEGLTRALRDGCFLLAVPPGFDTTPGVTLMREFFRPVEQGGEPTRAYRGFRDLDGVYFDREGFQKEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVLGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTAVVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"

    return  -nltk.edit_distance(originalSeq_, new_stateSeq_)



