import torch.nn as nn
from torch import manual_seed
from torch.distributions import Categorical

from typing import List

from copy import deepcopy

from src.residoraHelpers.convNet import convNet

class ActorCritic(nn.Module):
    """
    This class represents an Actor-Critic model. It takes in the following parameters:
    :param state_dim: An integer representing the dimensionality of the state space.
    :param action_dim: An integer representing the dimensionality of the action space.
    :param lr_actor: A float representing the learning rate for the actor.
    :param lr_critic: A float representing the learning rate for the critic.
    :param nrNeuronsInHiddenLayers: A list representing the number of neurons in each hidden layer.
    :param activationFunction: A string representing the activation function used in the model.
    :param seed: An integer representing the seed value for PyTorch.
    :param useCNN: A boolean indicating whether to use a convolutional neural network or not.
    :param stride: An integer representing the stride length used in the convolutional layer (if used).
    :param kernel_size: An integer representing the size of the kernel used in the convolutional layer (if used).
    :param out_channel: An integer representing the number of output channels from the convolutional layer (if used).
    :param dropOutProb: A float representing the probability of dropping out a neuron in the dropout layer.    
    """
    def __init__(self, state_dim : int, action_dim : int, multiAction : int = None, lr_actor : float = 0.0003, lr_critic : float = 0.001, 
                 nrNeuronsInHiddenLayers : List = [64, 64],  activationFunction : str = "tanh", seed : int = 13, 
                 useCNN = False, stride = 4, kernel_size = 8, out_channel = 4, dropOutProb : float = 0.05):
        super(ActorCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.multiAction = multiAction
        self.seed = seed
        self.useCNN = useCNN

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        #set seed
        manual_seed(self.seed)

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

            self.linearInputDim = int(state_dim/stride*out_channel)
            
            self.actor.cnn.append(
                nn.Sequential(
                    nn.Linear(self.linearInputDim , nrNeuronsInHiddenLayers[1]),
                    self.activationFunction(),
                    nn.Linear(nrNeuronsInHiddenLayers[1], action_dim),
                    nn.Softmax(dim = -1)               
                )
            )
            self.critic.cnn.append(
                nn.Sequential(
                    nn.Linear(self.linearInputDim , nrNeuronsInHiddenLayers[1]),
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

        try:
            #The probs of the actor the decide which one to mutate
            mutationProbabilities = self.actor(embedding)
        except Exception as err:
            print(err)
            print("-------------------------------------------------")
            print("Error at ActorCritic.select_action_exploration")
            print("-------------------------------------------------")
        #transform probs into probability distirbution
        dist = Categorical(mutationProbabilities)

        # TODO finish the multi Action approach
        if self.multiAction:
            actions = dist.sample_n(self.multiAction)
            actionLogProbs = dist.log_prob(actions)
            stateVal = self.critic(embedding)
            return actions.detach(), actionLogProbs.detach(), stateVal.detach()
        
        else: #single action
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

        try:
            #The probs of the actor the decide which one to mutate
            mutationProbabilities = self.actor(embedding)
        except Exception as err:
            print(err)
            print("-------------------------------------------------")
            print("Error at ActorCritic.evaluate")
            print("-------------------------------------------------")

        #transform probs into probability distirbution
        dist = Categorical(mutationProbabilities)

        actionLogProb   = dist.log_prob(action) # get the log prob of the decided action
        distEntropy     = dist.entropy()
        stateValues     = self.critic(embedding)

        return actionLogProb, stateValues, distEntropy