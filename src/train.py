import random
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV
import time
from collections import deque
from copy import deepcopy

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)

CONFIG = {'state_size':6,
          'action_size': env.action_space.n,
          'num_layers': 4,
          'nb_neurons': 512,
          'max_episode': 500,
          'learning_rate': 0.005,
          'gamma': 0.95,
          'buffer_capacity': 150000,
          'eps_min': 0.05,
          'eps_max': 1.,
          'eps_period': 20000,
          'batch_size': 1024,
          'criterion': torch.nn.MSELoss()}

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, CONFIG['nb_neurons'])
        for num_layers in range(1, CONFIG['num_layers']-1):
            setattr(self, f'fc{num_layers+1}', nn.Linear(CONFIG['nb_neurons'], CONFIG['nb_neurons']))
        setattr(self, f'fc{CONFIG['num_layers']}', nn.Linear(CONFIG['nb_neurons'], action_size))

    def forward(self, x):
        for num_layers in range(0, CONFIG['num_layers']-1):
            x = getattr(self, f'fc{num_layers+1}')(x)
            x = torch.relu(x)
        return getattr(self, f'fc{CONFIG['num_layers']}')(x)

class Buffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class ProjectAgent:

    def __init__(self, config=CONFIG):
        self.config = config
        self.q_network = QNetwork(config['state_size'], config['action_size'])
        self.target_network = deepcopy(self.q_network)
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config['learning_rate'])
        self.replay_buffer = Buffer(capacity=config['buffer_capacity'])

    def act(self, observation, use_random=False, epsilon=0.1):
        if use_random and random.uniform(0, 1) < epsilon:
            return random.randint(0, self.config['action_size'] - 1)
        else:
            with torch.no_grad():
                observation = torch.FloatTensor(observation).unsqueeze(0)
                q_values = self.q_network(observation)
                return torch.argmax(q_values).item()

    def save(self, path = 'best_model_3.pth'):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path='best_model_3.pth'):
        self.q_network.load_state_dict(torch.load(path))
        self.q_network.eval()

    def fit_q(self):
        if len(self.replay_buffer) < self.config['batch_size']:
            return
        batch = self.replay_buffer.sample(self.config['batch_size'])
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.LongTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch)
        q_values = self.q_network(state_batch)
        next_q_values = self.target_network(next_state_batch).detach()
        target_q_values = reward_batch + (1 - done_batch) * self.config['gamma'] * torch.max(next_q_values, dim=1)[0]
        loss = self.config['criterion'](q_values.gather(1, action_batch.unsqueeze(1)), target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        epsilon = self.config['eps_max']
        epsilon_step = (self.config['eps_max'] - self.config['eps_min']) / (self.config['eps_period'])
        episode = 0
        total_reward = 0
        best_validation = -np.inf
        state, _ = env.reset()
        n_step = 0
        start = time.time()
        while episode < self.config['max_episode']:
            epsilon = max(self.config['eps_min'], epsilon - epsilon_step)
            action = self.act(state, use_random=True, epsilon=epsilon)
            next_state, reward, done, trunc, _ = env.step(action)
            self.replay_buffer.push((state, action, reward, next_state, done))
            total_reward += reward
            self.fit_q()
            n_step += 1
            if done or trunc:
                validation_score = evaluate_HIV(agent=self, nb_episode=1)
                if validation_score > best_validation:
                    best_validation = validation_score
                    self.save()  
                state, _ = env.reset()
                end = time.time()
                print(f"Episode {episode + 1}/{self.config['max_episode']}, Total Reward: {total_reward}, Val Score: {validation_score}, Time: {np.round(end - start)}s")
                total_reward = 0  
                if (episode) % 50 == 0:
                    print(f"Best Evaluation Score for now: {best_validation}")
                start = time.time()
                episode += 1
            else:
                state = next_state
        print(f"Best Evaluation Score: {best_validation}")


if __name__ == "__main__":
    agent = ProjectAgent()
    agent.train()