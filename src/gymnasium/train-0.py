import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from typing import List, Tuple

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, action_dim)
        self.apply(self.init_weights)
        self.activation = nn.LeakyReLU()

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)
        self.apply(self.init_weights)
        self.activation = nn.LeakyReLU()

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

class ReplayMemory:
    def __init__(self, gamma: float, lambda_: float):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.memory = []

    def push(self, state, action, log_prob, reward, done, value):
        self.memory.append((state, action, log_prob, reward, done, value))

    def compute_returns_and_advantages(self, last_value):
        returns = []
        advantages = []
        gae = 0
        next_value = last_value

        for state, action, log_prob, reward, done, value in reversed(self.memory):
            if done:
                next_value = 0
            delta = reward + self.gamma * next_value - value
            gae = delta + self.gamma * self.lambda_ * gae
            next_value = value
            returns.insert(0, gae + value)
            advantages.insert(0, gae)

        return returns, advantages

    def clear_memory(self):
        self.memory = []

class PPOAgent:
    def __init__(self, env: gym.Env, policy_lr: float = 4e-4, value_lr: float = 2e-3, gamma: float = 0.99, clip_epsilon: float = 0.2, update_epochs: int = 10, gae_lambda: float = 0.95, entropy_coef: float = 0.005):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.value_network = ValueNetwork(self.state_dim).to(self.device)
        self.policy_optimizer = optim.RMSprop(self.policy_network.parameters(), lr=policy_lr, eps=1e-5)
        self.value_optimizer = optim.RMSprop(self.value_network.parameters(), lr=value_lr, eps=1e-5)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.MseLoss = nn.MSELoss()
        self.memory = ReplayMemory(gamma, gae_lambda)

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.policy_network.get_action(state)
        return action, log_prob

    def update(self):
        states, actions, log_probs, rewards, dones, values = zip(*self.memory.memory)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        old_log_probs = torch.cat(log_probs).to(self.device)
        last_value = self.value_network(states[-1])

        returns, advantages = self.memory.compute_returns_and_advantages(last_value)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.update_epochs):
            logits = self.policy_network(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions.squeeze())
            ratios = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            loss_entropy = self.entropy_coef * dist.entropy().mean()
            loss_value = 0.5 * self.MseLoss(returns, self.value_network(states).squeeze())
            loss_policy = -torch.min(surr1, surr2).mean() - loss_entropy

            self.policy_optimizer.zero_grad()
            loss_policy.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
            self.value_optimizer.step()

        self.memory.clear_memory()

    def train(self, max_episodes: int = 1000, max_timesteps: int = 700):
        episode_rewards = []
        for episode in range(max_episodes):
            state = self.env.reset()[0]
            episode_reward = 0

            for t in range(max_timesteps):
                action, log_prob = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                value = self.value_network(torch.FloatTensor(state).to(self.device)).item()
                self.memory.push(state, action, log_prob, reward, done, value)

                state = next_state
                episode_reward += reward

                if done:
                    break

            self.update()
            episode_rewards.append(episode_reward)
            print(f'Episode {episode + 1}/{max_episodes}, Reward: {episode_reward}')

        self.plot_rewards(episode_rewards)

    def evaluate(self, num_episodes: int = 10):
        self.env = gym.make("LunarLander-v2", render_mode="human")
        for episode in range(num_episodes):
            state = self.env.reset()[0]
            episode_reward = 0
            done = False
            while not done:
                self.env.render()
                action, _ = self.select_action(state)
                state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                if done:
                    print(f'Evaluation Episode {episode + 1}/{num_episodes} finished with reward: {episode_reward}')
                    break
        self.env.close()

    def plot_rewards(self, episode_rewards: List[float]):
        smoothed_rewards = np.convolve(episode_rewards, np.ones(20) / 20, mode='valid')
        plt.plot(smoothed_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('Episode Returns')
        plt.show()

# Create the environment and train the agent
env = gym.make("LunarLander-v2", enable_wind = True)
agent = PPOAgent(env)
agent.train(max_episodes=1000)
agent.evaluate(num_episodes=10)
