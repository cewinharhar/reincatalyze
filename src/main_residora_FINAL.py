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

import nltk


def prepare4APIRequest(payload : dict):
    package = {"predictionCallDict": payload}
    package["predictionCallDict"] = json.dumps(package["predictionCallDict"], 
                                               default=str, # use str method to serialize non-json data
                                               separators=(",", ":")) # remove spaces between separators
    return package

"""
In this function, we initialize the PPO agent with the specified hyperparameters, and start the training loop. 
In each iteration, we use the agent to select an action, modify the current state by randomly changing one element of the list, 
and calculate the corresponding reward. We then use the reward to update the agent's policy using the update method. The loop continues until the state no longer contains the "N" element.

Note that in this example, the reward_function is a simple function that assigns a reward of 1 if the chosen action results in the "N" element being replaced, 
and -1 otherwise. You can modify this function as needed to create more complex reward schemes based on the specific problem you are trying to solve.
"""

class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PPOAgent:
    def __init__(self, input_size, hidden_size, output_size, lr, gamma, eps_clip):
        self.policy = Policy(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        logits = self.policy(state)
        probs = torch.softmax(logits, dim=0)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), probs[action.item()].item()

    def update(self, states, actions, log_probs, rewards, dones):
        states = torch.from_numpy(np.array(states)).float()
        actions = torch.from_numpy(np.array(actions)).long()
        log_probs = torch.from_numpy(np.array(log_probs)).float()
        rewards = torch.from_numpy(np.array(rewards)).float()
        dones = torch.from_numpy(np.array(dones)).float()

        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8) #add small constant to avoid divition by 0

        old_log_probs = log_probs.detach()

        for i in range(10):
            logits = self.policy(states)
            probs = torch.softmax(logits, dim=1)
            dist = Categorical(probs)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(actions)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * returns
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * returns
            loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(returns, torch.zeros_like(returns))
            loss = loss.mean() - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def reward_function(originalSeq_, new_stateSeq_):
    # Calculate reward based on state and action
    # Here's a simple example:
    if isinstance(originalSeq_, list):
        originalSeq_ = "".join(originalSeq_)
    if isinstance(new_stateSeq_, list):
        new_stateSeq_ = "".join(new_stateSeq_)

    #TODO remove
    originalSeq_ = "MSSETTRLQNARATEECLAW"

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

def main(original = "MSTETLRLQKARATEEGLAF"):
    #TODO  TRY IF RL WORKS BY CHANGING THE APPROPIATE AMINO ACIDS TOGETHER WITH DEEPMUT AND EMBEDDING

    input_size = 1024
    hidden_size = 256
    output_size = len(list(original))
    lr = 0.001
    gamma = 0.99
    eps_clip = 0.2

    done = False

    agent = PPOAgent(input_size, hidden_size, output_size, lr, gamma, eps_clip)
    originalSeq = stateSeq = list(original)
    state = embeddingRequest(seq = stateSeq, returnNpArray = True)

    i = 0
    while not done:
        i += 1
        action, prob = agent.select_action(state)
        new_stateSeq = stateSeq.copy()

        #HERE COMES THE TRANSFORMER INPUT GAEPS
        new_stateSeq = list(deepMutRequest(seq = new_stateSeq, rationalMaskIdx=[action])[0])
        #new_stateSeq[action] = np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"))

        reward = reward_function(originalSeq_ = originalSeq, new_stateSeq_ = new_stateSeq)

        new_state = embeddingRequest(seq = new_stateSeq, returnNpArray=True)
        
        print(
            f"""------------ \n round:{i} \n------------ \n ori Seq: {"".join(originalSeq)} \n old Seq: {"".join(stateSeq)} \n new Seq: {"".join(new_stateSeq)} \n action: {str(action)} \n prob of action: {str(prob)} \n reward: {str(reward)}"""
        )

        state = new_state
        stateSeq = new_stateSeq

        agent.update([state], [action], [prob], [reward], [done])


        if i == 200:
            done = True

    print("Final state:", state)

main()

pprint("hi")