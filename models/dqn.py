import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm


class Qnet(torch.nn.Module):
    ''' A single-layer hidden Q network '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class Dueling_DQN:
    ''' DQN algorithm, including DoubleDQN '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device, dqn_type='VanillaDQN'):
        self.action_dim = action_dim 
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device) 
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device) 
        self.optimizer= torch.optim.Adam(self.q_net.parameters(), lr=learning_rate) 
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.target_update = target_update 
        self.count = 0
        self.dqn_type = dqn_type
    
    def take_action(self, state): 
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(device)
            action = self.q_net(state).argmax().item()
        return action
    
    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(device)
        actions =  torch.tensor(transition_dict['actions']).view(-1, 1).to(device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(device)

        q_values = self.q_net(states).gather(1, actions) 
        if self.dqn_type == 'DoubleDQN': # Double DQN
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1).to(device)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1) 
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones) 
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward() 
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

class SimpleBox:
    def __init__(self, low, high, shape=None, dtype=np.float32):
        self.low = np.array(low, dtype=dtype)
        self.high = np.array(high, dtype=dtype)
        if shape is not None:
            self.low = np.full(shape, self.low, dtype=dtype)
            self.high = np.full(shape, self.high, dtype=dtype)
        self.shape = self.low.shape
        self.dtype = dtype

    def sample(self):
        return np.random.uniform(low=self.low, high=self.high, size=self.shape).astype(self.dtype)

    def contains(self, x):
        return (x >= self.low).all() and (x <= self.high).all()