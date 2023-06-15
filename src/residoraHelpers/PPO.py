import torch
import torch.nn as nn
from copy import deepcopy
import os

class PPO:
    """
    The PPO class is used to perform the Proximal Policy Optimization (PPO) algorithm to update the policy of an actor-critic network.

    The class constructor initializes the class variables, including the buffer for storing the actions, states, log probabilities, rewards, state values, and isTerminal states. It also initializes the hyperparameters for PPO, such as the discount factor gamma, the number of epochs K_epochs for optimizing the policy, the clipping parameter eps_clip, and the device on which to perform computations. Additionally, it initializes the actor-critic network, the optimizer, and the old policy.

    The clearBuffer() function is used to clear the buffer after each training iteration.

    The select_action_exploitation(embedding) function is used to select the action with the highest probability or the highest expected value based on the learned policy of the agent during the exploitation phase. It accepts an embedding as input, and it returns the selected action.

    The update() function updates the policy of the PPO agent using the PPO update algorithm with the experience gathered in the replay buffer. It computes the Monte Carlo estimate of returns for the collected experiences, normalizes the rewards, and calculates the advantages. It then optimizes the policy for a fixed number of epochs using the PPO loss function, which is a clipped surrogate objective. Finally, it copies the new weights into the old policy, clears the replay buffer, and returns nothing.
        
    """
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
        print("PPO>select_action_exploitation")
        with torch.no_grad():
            embedding = torch.FloatTensor(embedding).to(self.device)
            action, actionLogProb, StateVal = self.policy_old.select_action_exploration(embedding)

        print(f"PPO>select_action_exploitation\n action: {action} \n actionLogProb: {actionLogProb} \n StateVal: {StateVal}" )
        #save the values in the buffer
        self.states.append(embedding)
        self.actions.append(action)
        self.logProbs.append(actionLogProb)
        self.stateValues.append(StateVal)
        return action.tolist()
    
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
        print("PPO>updating")
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

        if isinstance(self.actions[0], torch.Tensor):
            old_actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach().to(self.device)
        else:
            print("PPO>update>olc_actions multi action")
            old_actions = torch.stack([torch.stack(a) for a in self.actions]).detach().to(self.device)
            print(old_actions)

        old_logProbs = torch.squeeze(torch.stack(self.logProbs, dim=0)).detach().to(self.device)
        old_stateValues = torch.squeeze(torch.stack(self.stateValues, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_stateValues.detach()    

        #-----------------------------------------------------------------------------------------------
        # The PPO loss function calculation
        #-----------------------------------------------------------------------------------------------
        # Optimize the policy network for K epochs
        for kEpoch in range(self.K_epochs):
            print("PPO>kEpoch iteration")
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
        print("PPO>copyNewWeightsIntoOldPolicy")
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.clearBuffer()            

    def save(self, checkpoint_path):

        parent_directory = os.path.dirname(checkpoint_path)

        if not os.path.exists(parent_directory):
            os.makedirs(parent_directory)
            print(f"Created parent directory: {parent_directory}")
        else:
            print(f"Parent directory already exists: {parent_directory}")

        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
