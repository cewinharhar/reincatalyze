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

""" 
def prepare4APIRequest(payload : dict):
    package = {"predictionCallDict": payload}
    package["predictionCallDict"] = json.dumps(package["predictionCallDict"], 
                                               default=str, # use str method to serialize non-json data
                                               separators=(",", ":")) # remove spaces between separators
    return package """

"""
In this function, we initialize the PPO agent with the specified hyperparameters, and start the training loop. 
In each iteration, we use the agent to select an action, modify the current state by randomly changing one element of the list, 
and calculate the corresponding reward. We then use the reward to update the agent's policy using the update method. The loop continues until the state no longer contains the "N" element.

Note that in this example, the reward_function is a simple function that assigns a reward of 1 if the chosen action results in the "N" element being replaced, 
and -1 otherwise. You can modify this function as needed to create more complex reward schemes based on the specific problem you are trying to solve.
"""

#TODO remove PPO folders if necessary

class convNet(nn.Module):
    def __init__(self, activationFunction, out_channel, kernel_size : int, padding : int, stride : int, dropOutProb : float):
        super(convNet, self).__init__()
    
        self.cnn = torch.nn.Sequential(
                            torch.nn.Conv1d(in_channels = 1, out_channels = out_channel, kernel_size=kernel_size, padding=padding, stride = stride ), # 7x32
                            activationFunction(),
                            torch.nn.Dropout( dropOutProb ),
                            nn.Flatten()
        )

        #get the dimen

    def forward(self, x1D):
        #x3D = x1D.unsqueeze(0).unsqueeze(0)
        return self.cnn(x1D)
    


class ActorCritic(nn.Module):
    """ Class which inits the actor & critic, evaluates and acts for a discrete action space

        activationFunction = [Tanh, ReLu]
    """
    def __init__(self, state_dim : int, action_dim : int, lr_actor : float = 0.0003, lr_critic : float = 0.001, 
                 nrNeuronsInHiddenLayers : List = [64, 64],  activationFunction : str = "tanh", seed : int = 13, 
                 useCNN = False, stride = 4, kernel_size = 8, out_channel = 4, dropOutProb : float = 0.05):
        super(ActorCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seed = seed
        self.useCNN = useCNN

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        #set seed
        torch.manual_seed(self.seed)

        #set activation function 
        if activationFunction.lower() == "relu":
            self.activationFunction = nn.ReLU
        elif activationFunction.lower() == "leakyrelu":
            self.activationFunction = nn.LeakyReLU
        elif activationFunction.lower() == "tanh":
            self.activationFunction = nn.Tanh

        if useCNN:
            self.cnn = convNet(activationFunction = self.activationFunction, out_channel = out_channel, kernel_size=kernel_size, padding = 1, stride = stride, dropOutProb=dropOutProb)
            
            self.actor = deepcopy(self.cnn)
            self.critic = deepcopy(self.cnn)

            linearInputDim = int(state_dim/stride*out_channel)
            
            self.actor.cnn.append(
                nn.Sequential(
                    nn.Linear(linearInputDim , nrNeuronsInHiddenLayers[1]),
                    self.activationFunction(),
                    nn.Linear(nrNeuronsInHiddenLayers[1], action_dim),
                    nn.Softmax(dim = -1)               
                )
            )
            self.critic.cnn.append(
                nn.Sequential(
                    nn.Linear(linearInputDim , nrNeuronsInHiddenLayers[1]),
                    self.activationFunction(),
                    nn.Linear(nrNeuronsInHiddenLayers[1], 1)           
                )
            )

        else:      
            #create actor and critic
            self.actor = nn.Sequential(
                nn.Linear(state_dim, nrNeuronsInHiddenLayers[0]),
                self.activationFunction(), #saves memory
                nn.Dropout(dropOutProb),
                nn.Linear(nrNeuronsInHiddenLayers[0], nrNeuronsInHiddenLayers[1]),
                self.activationFunction(), #saves memory
                nn.Dropout(dropOutProb),
                nn.Linear(nrNeuronsInHiddenLayers[1], action_dim),
                nn.Softmax(dim = -1)
            )
            #the critic has only 1 output node which is the estimation of how well the actor has decided
            self.critic = nn.Sequential(
                nn.Linear(state_dim, nrNeuronsInHiddenLayers[0]),
                self.activationFunction(), #saves memory
                nn.Dropout(dropOutProb),
                nn.Linear(nrNeuronsInHiddenLayers[0], nrNeuronsInHiddenLayers[1]),
                self.activationFunction(), #saves memory
                nn.Dropout(dropOutProb),
                nn.Linear(nrNeuronsInHiddenLayers[1], 1)
            )

    def select_action_exploration(self, embedding_):  #also called select_action
        """ This function takes in the embedding and makes a decision on which residue to mutate
            The act function is used during training to select actions based on the current state of the environment, as well as the policy learned so far. 
            It takes as input a state tensor and returns three tensors:
                - action: A tensor containing the selected actions. The tensor is detached from the computation graph to avoid backpropagation through it.
                - action_logprob: A tensor containing the log-probabilities of the selected actions under the policy. This tensor is used later in the PPO loss function to encourage the policy to select actions with high probability.
                - state_val: A tensor containing the estimated state value of the current state under the critic network. This tensor is used in the computation of the PPO loss function.

            The select_action() function without torch.no_grad() is used during the EXPLORATION phase, 
            where the agent needs to generate a new action based on its current policy. 
            This function calculates the action probabilities or the mean and variance of the action distribution, 
            depending on the type of action space, and then samples a new action from the distribution.
        """
        if self.useCNN:
            embedding = embedding_.unsqueeze(0).unsqueeze(0)
        else:
            embedding = embedding_

        #The probs of the actor the decide which one to mutate
        mutationProbabilities = self.actor(embedding)
        #transform probs into probability distirbution
        dist = Categorical(mutationProbabilities)

        action = dist.sample() #take one action of the prob distribution
        actionLogProb = dist.log_prob(action) # get the log prob of the decided action
        #get critis opinion
        stateVal = self.critic(embedding)

        return action.detach(), actionLogProb.detach(), stateVal.detach()
    
    def evaluate(self, embedding_, action):  #also called select_action
        """ 
            The evaluate function, on the other hand, is used to compute the log-probability of a given action under the current policy, 
            as well as other quantities that are used in the computation of the PPO loss function. 
            It takes as input a state tensor and an action tensor, and returns three tensors:
                - action_logprobs: A tensor containing the log-probabilities of the given actions under the policy. This tensor is used in the computation of the PPO loss function to encourage the policy to select actions with high probability.
                - dist_entropy: A tensor containing the entropy of the action distribution. This term is included in the PPO loss function to encourage exploration, by penalizing policies that are too deterministic.
                - state_values: A tensor containing the estimated state value of the current state under the critic network. This tensor is used in the computation of the PPO loss function.
        """
        if self.useCNN:
            #permutation to tell CNN that we have batch size of 3 but still only 1 channel
            embedding   = embedding_.unsqueeze(0).permute([1,0,2])
        else:
            embedding   = embedding_

        #The probs of the actor the decide which one to mutate
        mutationProbabilities = self.actor(embedding) 
        #transform probs into probability distirbution
        dist = Categorical(mutationProbabilities)

        actionLogProb   = dist.log_prob(action) # get the log prob of the decided action
        distEntropy     = dist.entropy()
        stateValues     = self.critic(embedding)

        return actionLogProb, stateValues, distEntropy
    

class PPO:
    """
    This class selects the mutations and updates the policy of actor and critic"""
    def __init__(self, ActorCritic, gamma = 0.99, K_epochs = 40, eps_clip = 0.2, device = None):

        if not device:
            if(torch.cuda.is_available()): 
                self.device = torch.device('cuda:0') 
                torch.cuda.empty_cache()
                print("Device set to : " + str(torch.cuda.get_device_name(self.device)))
            else:
                print("Device set to : cpu")  
        else:
            self.device = device

        #storage for values in Buffer
        self.actions = []
        self.states = []
        self.logProbs = []
        self.rewards = []
        self.stateValues = []
        self.isTerminals = []
        
        #hyperparameters 
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        #init actor & critic
        self.policy = ActorCritic.to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.policy.lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': self.policy.lr_critic}
                    ])
        
        #reinitialize actor and critic to use as comparison
        #The reason for reinitializing the actor and critic in PPO is to create a separate copy of the policy network, 
        # which is used to calculate the policy loss during the training process. 
        # This copy is referred to as the "old policy", or the "previous policy".
        #Reinitialize the actor and critic
        self.policy_old = deepcopy(self.policy)
        #copy parameters from the first actor & critic network
        self.policy_old.load_state_dict(self.policy.state_dict())        

        self.MseLoss = nn.MSELoss()

    def clearBuffer(self):
        del self.actions[:]
        del self.states[:]
        del self.logProbs[:]
        del self.rewards[:]
        del self.stateValues[:]
        del self.isTerminals[:]        

    def select_action_exploitation(self, embedding):
        """
        The select_action() function with torch.no_grad() is used during the EXPLOITATION phase, 
        where the agent needs to select the action that maximizes the expected return based on its learned policy. 
        This function runs the policy network forward with the torch.no_grad() context, 
        which disables gradient calculations and speeds up the computation. 
        It then selects the action with the highest probability or the highest expected value, 
        depending on the action space, using the argmax() function.
        """

        with torch.no_grad():
            embedding = torch.FloatTensor(embedding).to(self.device)
            action, actionLogProb, StateVal = self.policy_old.select_action_exploration(embedding)

        #save the values in the buffer
        self.states.append(embedding)
        self.actions.append(action)
        self.logProbs.append(actionLogProb)
        self.stateValues.append(StateVal)

        return action.item()
    
    def update(self):
        """
            The update() function updates the policy of the PPO agent by performing the PPO update algorithm with the experience gathered in the replay buffer.

            The function computes the Monte Carlo estimate of returns for the collected experiences, normalizes the rewards, and calculates the advantages. 
            It then optimizes the policy for a fixed number of epochs using the PPO loss function, which is a clipped surrogate objective. 
            Finally, it copies the new weights into the old policy, clears the replay buffer, and returns nothing.

            Args:
            None

            Returns:
            None        
        """

        rewards = []
        discounted_reward = 0

        
        #reversed; the discounted reward calculation involves looking ahead to future rewards, 
        # so starting from the end of the list allows for the cumulative sum of discounted rewards to be computed in a singlepass.
        for reward, isTerminal in zip(reversed(self.rewards), reversed(self.isTerminals)):     
            if isTerminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach().to(self.device)
        old_logProbs = torch.squeeze(torch.stack(self.logProbs, dim=0)).detach().to(self.device)
        old_stateValues = torch.squeeze(torch.stack(self.stateValues, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_stateValues.detach()    

        #-----------------------------------------------------------------------------------------------
        # The PPO loss function calculation
        #-----------------------------------------------------------------------------------------------
        # Optimize the policy network for K epochs
        for kEpoch in range(self.K_epochs):
            #print(kEpoch)
            # Evaluating old actions and values
            #print("Update > evaluate")
            #print(old_states)
            #print(old_states.shape)
            logProbs, stateValues, distEntropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            #print("Update > sequeeze")
            stateValues = torch.squeeze(stateValues)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logProbs - old_logProbs.detach())

            # Finding Surrogate Loss   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(stateValues, rewards) - 0.01 * distEntropy
            
            # take gradient step
            #print("Update > grad Step")
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        #-----------------------------------------------------------------------------------------------
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.clearBuffer()            

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

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



def embeddingRequest(seq, returnNpArray = False, embeddingUrl = "http://0.0.0.0/embedding"):
    #check if list, if not, make oen
    #if not isinstance(seq, list):
    #    seq = [seq]
    seq = ["".join(seq)]    

    payload = dict(inputSeq = seq)

    package = prepare4APIRequest(payload)

    try:
        response = requests.post(embeddingUrl, json=package).content.decode("utf-8")
        embedding = json.loads(response) 
        if returnNpArray:   
            return np.array(embedding[0])
        else:
            return embedding[0]
    
    except requests.exceptions.RequestException as e:
        errMes = "Something went wrong\n" + str(e)
        print(errMes)
        raise Exception
    
f = embeddingRequest(seq = "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA", returnNpArray=True)
s = embeddingRequest(seq = "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQVGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA", returnNpArray=True)

import matplotlib.pyplot as plt

def deepMutRequest(seq, rationalMaskIdx):
    deepMutUrl = "http://0.0.0.0/deepMut"
    nrOfSequences = 1
    #INIT WITH WILDTYPE
    payload = dict(
                        inputSeq            = [x for x in seq],
                        task                = "rational",
                        rationalMaskIdx     = rationalMaskIdx ,
                        huggingfaceID       = "Rostlab/prot_t5_xl_uniref50",
                        num_return_sequences= nrOfSequences,
                        max_length          = 512,
                        do_sample           = True,
                        temperature         = 1.5,
                        top_k               = 20
    )
    #make json
    package = prepare4APIRequest(payload)

    #get predictions
    try:
        response = requests.post(deepMutUrl, json=package).content.decode("utf-8")
        deepMutOutput = json.loads(response)
        return deepMutOutput

    except requests.exceptions.RequestException as e:
        errMes = "Something went wrong\n" + str(e)
        print(errMes)
        pass

#⨪-----------------------------------

aKGD31 = "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"

main( original = aKGD31)

