import numpy as np
import torch
import gymnasium as gym
from matplotlib import pyplot as plt
import display

env = gym.make('CartPole-v1')

obs_size = env.observation_space.shape
n_actions = env.action_space.n  
HIDDEN_SIZE = 256

model = torch.nn.Sequential(
             torch.nn.Linear(obs_size, HIDDEN_SIZE), #4 observations, 256 hidden
             torch.nn.ReLU(),
             torch.nn.Linear(HIDDEN_SIZE, n_actions), #256 hidden, 2 output (left, right)
             torch.nn.Softmax(dim=0)
     )

#define optimizer and init param
learning_rate = 0.003 #= step size alpha
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
Horizon = 500 #how many trials
MAX_TRAJECTORIES = 500
gamma = 0.99
score = []

#................................................
for trajectory in range(MAX_TRAJECTORIES):
    curr_state = env.reset()
    done = False
    transitions = [] 
    
    for t in range(Horizon):
        act_prob = model(torch.from_numpy(curr_state).float())
        action = np.random.choice(np.array([0,1]), p=act_prob.data.numpy())
        prev_state = curr_state
        curr_state, _, done, info = env.step(action)
        transitions.append((prev_state[0], action, t+1)) 
        if done: 
            break
    score.append(len(transitions))

    reward_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=(0,))     
  
    batch_Gvals =[]
    for i in range(len(transitions)):
        new_Gval=0
        power=0
        for j in range(i,len(transitions)):
             new_Gval=new_Gval+((gamma**power)*reward_batch[j]).numpy()
             power+=1
        batch_Gvals.append(new_Gval)
    expected_returns_batch=torch.FloatTensor(batch_Gvals)
    expected_returns_batch /= expected_returns_batch.max()    
    state_batch = torch.Tensor([s for (s,a,r) in transitions]) 
    action_batch = torch.Tensor([a for (s,a,r) in transitions])     
    pred_batch = model(state_batch) 
    prob_batch = pred_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze() 
    
    loss= -torch.sum(torch.log(prob_batch)*expected_returns_batch) 
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if trajectory % 50 == 0 and trajectory>0:
        print('Trajectory {}\tAverage Score: {:.2f}'
                .format(trajectory, np.mean(score[-50:-1])))
#................................................
