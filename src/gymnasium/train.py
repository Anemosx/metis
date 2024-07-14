import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import Env
from torch.distributions import Categorical


class DenseNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 1):
        super(DenseNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, action_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


class ReplayMemory:
    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95, adjust_rewards: bool = True, device: str | torch.device | None = "cpu"):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.adjust_rewards = adjust_rewards
        self.memory = []

    def push(self, state, action, log_prob, reward, done, value):
        if self.adjust_rewards:
            reward = reward - 0.3

        self.memory.append((state, action, log_prob, reward, done, value))

    def create_batch(self):
        states, actions, log_probs, rewards, dones, values = zip(*self.memory)
        old_log_probs = torch.cat(log_probs)

        states = torch.stack(states)
        actions = torch.stack(actions).unsqueeze(1)

        self.memory = []

        return states, actions, rewards, dones, values, old_log_probs


def compute_returns_advantages(last_value, rewards, dones, values, gamma: float = 0.99, gae_lambda: float = 0.95, device: str | torch.device | None = "cpu"):
    returns = []
    advantages = []
    gae = 0
    next_value = last_value

    for reward, done, value in reversed(list(zip(rewards, dones, values))):
        if done:
            next_value = 0
        delta = reward + gamma * next_value - value
        gae = delta + gamma * gae_lambda * gae
        next_value = value
        returns.insert(0, gae + value)
        advantages.insert(0, gae)

    returns = torch.tensor(returns, device=device)
    advantages = torch.tensor(advantages, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages


class PPOAgent:
    def __init__(self, env: gym.Env, policy_lr: float = 4e-4, value_lr: float = 2e-3, opt_eps: float = 1e-6,
                 clip_eps: float = 0.2, clip_grad: float = 1, update_epochs: int = 10, entropy_coef: float = 0.001, device: str | torch.device | None = "cpu"):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.device = device
        self.policy_net = DenseNet(self.state_dim, self.action_dim).to(self.device)
        self.value_net = DenseNet(self.state_dim).to(self.device)
        self.policy_optimizer = optim.RMSprop(self.policy_net.parameters(), lr=policy_lr, eps=opt_eps)
        self.value_optimizer = optim.RMSprop(self.value_net.parameters(), lr=value_lr, eps=opt_eps)
        self.clip_eps = clip_eps
        self.clip_grad = clip_grad
        self.update_epochs = update_epochs
        self.entropy_coef = entropy_coef
        self.MseLoss = nn.MSELoss()

    def select_action(self, state: np.ndarray | torch.Tensor, estimate_value: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device)
        with torch.no_grad():
            logits = self.policy_net(state.unsqueeze(0))
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        value = None
        if estimate_value:
            value = self.value_net(state)
        return action, log_prob, value

    def update(self, states: torch.Tensor, actions: torch.Tensor, rewards, dones, values, old_log_probs: torch.Tensor):

        last_value = self.value_net(states[-1])

        returns, advantages = compute_returns_advantages(last_value, rewards, dones, values, gamma=0.99, gae_lambda=0.95, device=self.device)

        for _ in range(self.update_epochs):
            logits = self.policy_net(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions.squeeze())
            ratios = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

            loss_entropy = self.entropy_coef * dist.entropy().mean()
            loss_value = 0.5 * self.MseLoss(returns, self.value_net(states).squeeze())
            loss_policy = -torch.min(surr1, surr2).mean() - loss_entropy

            self.policy_optimizer.zero_grad()
            loss_policy.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_grad)
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.clip_grad)
            self.value_optimizer.step()


def train_model(env: Env, agent, memory, max_episodes: int = 1000, max_steps: int = 700, smooth_window: int = 20):
    episode_rewards = []
    for episode in range(max_episodes):
        state = env.reset()[0]
        state = torch.tensor(state, device=agent.device)
        episode_reward = 0

        for t in range(max_steps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action.item())
            memory.push(state, action, log_prob, reward, done, value)

            state = torch.tensor(next_state, device=agent.device)
            episode_reward += reward

            if done:
                break

        agent.update(*memory.create_batch())
        episode_rewards.append(episode_reward)
        print(f'Episode {episode + 1}/{max_episodes}, Reward: {episode_reward}')

    smoothed_rewards = np.convolve(episode_rewards, np.ones(smooth_window) / smooth_window, mode='valid')
    plt.plot(smoothed_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Episode Returns')
    plt.show()


def evaluate_model(env: Env, agent, num_episodes: int = 10, max_steps: int = 800):
    for episode in range(num_episodes):
        state = env.reset()[0]
        episode_reward = 0
        for t in range(max_steps):
            env.render()
            action, _, _ = agent.select_action(state, estimate_value=False)
            state, reward, done, _, _ = env.step(action.item())
            episode_reward += reward
            if done:
                print(
                    f'Evaluation Episode {episode + 1}/{num_episodes} finished with reward: {episode_reward}')
                break
    env.close()


def run() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gamma = 0.99
    gae_lambda = 0.95

    max_episodes = 1000
    max_steps = 700

    eval_num_episodes = 10
    eval_max_steps = 800

    train_env: Env = gym.make("LunarLander-v2", enable_wind=True, wind_power=5)
    eval_env: Env = gym.make("LunarLander-v2", render_mode="human")

    agent = PPOAgent(train_env, device=device)

    memory = ReplayMemory(gamma, gae_lambda, device=device)

    train_model(train_env, agent, memory, max_episodes=max_episodes, max_steps=max_steps, smooth_window=10)

    evaluate_model(eval_env, agent, num_episodes=eval_num_episodes, max_steps=eval_max_steps)


if __name__ == "__main__":
    run()
