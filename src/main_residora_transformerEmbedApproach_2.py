import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

"""
In this implementation, the Policy class defines a simple neural network policy with two fully connected layers, 
and the ReinforcementLearningAgent class encapsulates the agent's policy and optimization algorithm.
The select_action method takes a state as input and returns an action selected from the policy based on the given state.
The update_policy method takes a list of rewards and corresponding log probabilities of actions, 
and computes the gradient of the expected cumulative reward with respect to the policy parameters using the policy gradient method. 
The gradient is then used to update the policy parameters using the Adam optimizer.Finally, the main method runs a fixed number of episodes, 
where each episode consists of selecting actions based on the current policy, receiving rewards from the environment, 
and updating the policy parameters based on the received rewards and log probabilities.Note that in this example, 
the environment is simulated by randomly changing the values of certain positions in a random vector

"""
class PolicyREINFORCE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyREINFORCE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x
    
class PolicyTRANSFORMER(nn.Module):
    def __init__(self, transformer_model, hidden_dim, output_dim):
        super(PolicyTRANSFORMER, self).__init__()
        self.transformer = transformer_model
        self.fc1 = nn.Linear(transformer_model.config.hidden_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.transformer(x).last_hidden_state[:, 0, :]
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x
    


class RLAgent:
    def __init__(self, Policy, input_dim, hidden_dim, output_dim, learning_rate, gamma):
        self.policy = Policy(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        action = torch.multinomial(probs, 1)
        return action.item()
    def update_policy(self, rewards, log_probs):
        discounted_rewards = []
        R = 0
        for r in rewards[::-1]:
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
            
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
            self.optimizer.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            self.optimizer.step()

def main(Policy = PolicyREINFORCE):
    # Define the environment
    input_dim = 10
    output_dim = 4
    # Define the agent
    agent = RLAgent(Policy = Policy, input_dim = input_dim, hidden_dim = 64, output_dim = output_dim, learning_rate = 1e-2, gamma = 0.99)
    # Run the episodes
    num_episodes = 1000

    for episode in range(num_episodes):
        state = np.random.rand(input_dim)
        rewards = []
        log_probs = []
        for step in range(20):
            action = agent.select_action(state)
            state[action] += np.random.normal(0, 0.1)
            reward = np.random.rand(2)
            rewards.append(reward)
            log_probs.append(torch.log(agent.policy(torch.from_numpy(state).float())[action]))
        agent.update_policy(rewards, log_probs)


main(Policy = PolicyREINFORCE)

if __name__ == '__main__':
    main()



#------------------------------------------

input_dim = 10
output_dim = 4

agent = RLAgent(Policy = PolicyREINFORCE, input_dim = input_dim, hidden_dim = 64, output_dim = output_dim, learning_rate = 1e-2, gamma = 0.99)

state = np.random.rand(input_dim)