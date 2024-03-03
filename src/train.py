import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import pickle
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import time

env_hiv = TimeLimit(HIVPatient(), max_episode_steps=200)

def action_selection(Q, state, action_count):
    Q_values = [Q.predict(np.append(state, a).reshape(1, -1)) for a in range(action_count)]
    return np.argmax(Q_values)

class ProjectAgent:
    def __init__(self, env=env_hiv, max_samples=None, max_iterations=None):
        self.env = env
        self.max_samples = max_samples
        self.max_iterations = max_iterations
        self.Q_values = []
        self.best_reward = -np.inf

    def gather_samples(self, horizon):
        state, _ = self.env.reset()
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        for i in range(horizon):
            action = self.act(state, random_action=True)
            next_state, reward, done, trunc, _ = self.env.step(action)
            if len(self.states) >= self.max_samples:
                self.states.pop(0)
                self.actions.pop(0)
                self.rewards.pop(0)
                self.next_states.pop(0)
                self.dones.pop(0)
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
            state = next_state if not (done or trunc) else self.env.reset()[0]
            if (i + 1) % 100 == 0:
                print(f"Gathered {i + 1} samples")

    def train(self, gamma, start=True):
        state_action_pairs = np.hstack((np.array(self.states), np.array(self.actions).reshape(-1, 1)))
        for i in range(self.max_iterations):
            start_time = time.time()
            if start:
                targets = self.rewards.copy()
            else:
                Q_next = np.zeros((len(self.states), self.env.action_space.n))
                for a in range(self.env.action_space.n):
                    next_state_action_pairs = np.hstack((self.next_states, a * np.ones((len(self.states), 1))))
                    Q_next[:, a] = self.Q_values[-1].predict(next_state_action_pairs)
                targets = np.array(self.rewards) + gamma * (1 - np.array(self.dones)) * np.max(Q_next, axis=1)
            Q = ExtraTreesRegressor(n_estimators=50, max_depth=10)
            Q.fit(state_action_pairs, targets)
            self.Q_values.append(Q)
            episode_reward = self.evaluate()
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.save('best_model.pkl')
            end_time = time.time()
            print(f"Iteration {i + 1}, Reward: {np.round(episode_reward)}, Time: {np.round(end_time - start_time)}s")

    def evaluate(self):
        state, _ = self.env.reset()
        total_reward = 0
        for _ in range(200):
            action = self.act(state)
            state, reward, done, trunc, _ = self.env.step(action)
            total_reward += reward
            if done or trunc:
                break
        return total_reward
    
    def act(self, state: np.ndarray, random_action: bool = False) -> int:
        if random_action and np.random.rand() < 0.1 or not self.Q_values:
            return self.env.action_space.sample()
        else:
            return action_selection(self.Q_values[-1], state, self.env.action_space.n)

    def save(self, file_path: str) -> None:
        with open(file_path, 'wb') as f:
            pickle.dump(self.Q_values[-1], f)

    def load(self) -> None:
        with open('best_model.pkl', 'rb') as f:
            self.Q_values = [pickle.load(f)]

if __name__ == '__main__':

    gamma = 0.98
    max_samples = 10000
    max_iterations = 100
    horizon = 1000
    agent = ProjectAgent(env_hiv, max_samples, max_iterations)
    agent.gather_samples(horizon)
    agent.train(gamma)