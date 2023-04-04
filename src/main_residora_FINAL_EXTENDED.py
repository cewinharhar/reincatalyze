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

class ActorCritic(nn.Module):
    """ Class which inits the actor & critic, evaluates and acts for a discrete action space

        activationFunction = [Tanh, ReLu]
    """

    def __init__(self, state_dim, action_dim, activationFunction = "tanh", seed = 13, nrNeuronsInHiddenLayers = [512, 256], dropOutProb = 0.2):
        super(ActorCritic, self).__init__()

        self.state_dim
        self.action_dim
        self.seed

        #set seed
        torch.manual_seed(self.seed)

        #set activation function 
        if activationFunction.lower() == "relu":
            self.activationFunction = nn.ReLU
        elif activationFunction.lower() == "tanh":
            self.activationFunction = nn.Tanh

        #create actor and critic
        self.actor = nn.Sequential(
            nn.Linear(state_dim, nrNeuronsInHiddenLayers[0]),
            self.activationFunction(),
            nn.Dropout(dropOutProb),
            nn.Linear(nrNeuronsInHiddenLayers[0], nrNeuronsInHiddenLayers[1]),
            self.activationFunction(),
            nn.Dropout(dropOutProb),
            nn.Linear(nrNeuronsInHiddenLayers[0], action_dim),
            nn.Softmax(dim = -1)
        )
        #the critic has only 1 output node which is the estimation of how well the actor has decided
        self.critic = nn.Sequential(
            nn.Linear(state_dim, nrNeuronsInHiddenLayers[0]),
            self.activationFunction(),
            nn.Dropout(dropOutProb),
            nn.Linear(nrNeuronsInHiddenLayers[0], nrNeuronsInHiddenLayers[1]),
            self.activationFunction(),
            nn.Dropout(dropOutProb),
            nn.Linear(nrNeuronsInHiddenLayers[0], 1)
        )

    def select_action_exploration(self, embedding):  #also called select_action
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
        #The probs of the actor the decide which one to mutate
        mutationProbabilities = self.actor(embedding)
        #transform probs into probability distirbution
        dist = Categorical(mutationProbabilities)

        action = dist.sample() #take one action of the prob distribution
        actionLogProb = dist.log_prob(action) # get the log prob of the decided action
        #get critis opinion
        stateVal = self.critic(embedding)

        return action.detach(), actionLogProb.detach(), stateVal.detach()
    
    def evaluate(self, embedding, action):  #also called select_action
        """ 
            The evaluate function, on the other hand, is used to compute the log-probability of a given action under the current policy, 
            as well as other quantities that are used in the computation of the PPO loss function. 
            It takes as input a state tensor and an action tensor, and returns three tensors:
                - action_logprobs: A tensor containing the log-probabilities of the given actions under the policy. This tensor is used in the computation of the PPO loss function to encourage the policy to select actions with high probability.
                - dist_entropy: A tensor containing the entropy of the action distribution. This term is included in the PPO loss function to encourage exploration, by penalizing policies that are too deterministic.
                - state_values: A tensor containing the estimated state value of the current state under the critic network. This tensor is used in the computation of the PPO loss function.
        """
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
    def __init__(self, state_dim, action_dim, lr_actor = 0.0003, lr_critic = 0.001, gamma = 0.99, K_epochs = 40, eps_clip = 0.2, device = None):

        if not device:
            if(torch.cuda.is_available()): 
                self.device = torch.device('cuda:0') 
                torch.cuda.empty_cache()
                print("Device set to : " + str(torch.cuda.get_device_name(self.device)))
            else:
                print("Device set to : cpu")  

        #storage for values in Buffer
        self.actions = []
        self.states = []
        self.logProbs = []
        self.rewards = []
        self.stateValues = []
        self.isTerminal = []
        
        #hyperparameters 
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        #init actor & critic
        self.policy = ActorCritic(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        
        #reinitialize actor and critic to use as comparison
        #The reason for reinitializing the actor and critic in PPO is to create a separate copy of the policy network, 
        # which is used to calculate the policy loss during the training process. 
        # This copy is referred to as the "old policy", or the "previous policy".
        #Reinitialize the actor and critic
        self.policy_old = ActorCritic(state_dim=state_dim, action_dim=action_dim).to(self.device)
        #copy parameters from the first actor & critic network
        self.policy_old.load_state_dict(self.policy.state_dict())        

        self.MseLoss = nn.MSELoss()

    def clearBuffer(self):
        del self.actions[:]
        del self.states[:]
        del self.logProbs[:]
        del self.rewards[:]
        del self.stateValues[:]
        del self.isTerminal[:]        

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
            state = torch.FloatTensor(state).to(self.device)
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
        for reward, isTerminal in zip(reversed(self.rewards), reversed(self.isTerminal)):     
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
         
            # Evaluating old actions and values
            logProbs, stateValues, distEntropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            stateValues = torch.squeeze(stateValues)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logProbs - old_logProbs.detach())

            # Finding Surrogate Loss   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(stateValues, rewards) - 0.01 * distEntropy
            
            # take gradient step
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



######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################


class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seed = 13):
        super(Policy, self).__init__()
        #set seed
        torch.manual_seed(seed)
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

        #normalize 
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

def main(original = "MSTETLRLQKARATEEGLAF", generations = 1000):
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


        if i == generations or reward == 0:
            done = True

    print("Final state:", state)


#⨪-----------------------------------

aKGD31 = "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"

main( original = aKGD31)

pprint("hi")