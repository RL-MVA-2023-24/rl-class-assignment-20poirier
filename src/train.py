from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time
import numpy as np

MAX_EPISODE_STEP = 200
NUM_EPISODES = 20

env_hiv = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=MAX_EPISODE_STEP
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

class ProjectAgent:
    def __init__(self, state_size=6, action_size=4, batch_size=64, gamma=0.9):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer(capacity=10000)

    def act(self, observation, use_random=True, epsilon=0.1):
        if use_random and random.uniform(0, 1) < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                observation = torch.FloatTensor(observation).unsqueeze(0)
                q_values = self.q_network(observation)
                return torch.argmax(q_values).item()

    def save(self, path = 'model.pth'):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path='model.pth'):
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def train(self, state, action, reward, next_state, done):
        self.replay_buffer.push((state, action, reward, next_state, done))
        if len(self.replay_buffer.buffer) > self.batch_size:
            batch = self.replay_buffer.sample(self.batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
            state_batch = torch.FloatTensor(state_batch)
            action_batch = torch.LongTensor(action_batch)
            reward_batch = torch.FloatTensor(reward_batch)
            next_state_batch = torch.FloatTensor(next_state_batch)
            done_batch = torch.FloatTensor(done_batch)
            q_values = self.q_network(state_batch)
            next_q_values = self.target_network(next_state_batch).detach()
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * torch.max(next_q_values, dim=1)[0]
            loss = nn.MSELoss()(q_values.gather(1, action_batch.unsqueeze(1)), target_q_values.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update(self.q_network, self.target_network)

    def update(self, local_model, target_model, tau=0.001):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def train_agent(agent, env, num_episodes=NUM_EPISODES):
    for episode in range(num_episodes):
        start = time.time()
        observation = env.reset()[0]
        total_reward = 0.0
        for n_step in range(MAX_EPISODE_STEP):
            eps = max(0.1, 0.9 - 0.9 * n_step / MAX_EPISODE_STEP)
            action = agent.act(observation, epsilon=eps)
            new_observation, reward, done, _, _ = env.step(action)
            agent.train(observation, action, reward, new_observation, done)
            observation = new_observation
            total_reward += reward
            if done:
                break
        end = time.time()
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Time: {np.round(end - start)}s")

if __name__ == "__main__":

    state_size = 6
    action_size = 4
    batch_size = 64
    gamma = 0.9

    dqn_agent = ProjectAgent(state_size, action_size, batch_size, gamma)

    train_agent(dqn_agent, env_hiv)

    dqn_agent.save()