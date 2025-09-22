from collections import deque
import torch 
import random
import numpy as np 


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.int32),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            np.array(next_states, dtype=np.int32),
            torch.BoolTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)