import os

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import Env
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class DenseNet(nn.Module):
    """
    Fully connected Neural Network for reinforcement learning.

    Parameters
    ----------
    state_dim : int
        The dimensionality of the input state space.
    action_dim : int, optional
        The dimensionality of the output action space (default is 1).

    Methods
    -------
    forward(inputs)
        Forward pass through the network.
    """

    def __init__(self, state_dim: int, action_dim: int = 1):
        """
        Initialize the Fully connected Neural Network model.

        This constructor initializes the fully connected layers and leaky relu
        activation function.

        Parameters
        ----------
        state_dim : int
            The dimensionality of the input state space.
        action_dim : int, optional
            The dimensionality of the output action space (default is 1).
        """

        super(DenseNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, action_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the network.

        Parameters
        ----------
        inputs : torch.Tensor
            A batch of input states [batch_size, state_dim], where `batch_size`
            is the number of observations in the batch.

        Returns
        -------
        out : torch.Tensor
            The output logits for each action of shape [batch_size, action_dim].
        """

        x = self.activation(self.fc1(inputs))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        out = self.fc4(x)
        return out


class Memory:
    """
    A memory buffer for storing experience tuples and computing returns and advantages
    for reinforcement learning algorithms. It supports optional reward adjustment and
    uses Generalized Advantage Estimation (GAE) for calculating the advantages.

    Parameters
    ----------
    gamma : float, optional
        The discount factor for future rewards (default is 0.99).
    gae_lambda : float, optional
        The smoothing parameter for Generalized Advantage Estimation (default is 0.95).
    adjust_rewards : bool, optional
        Whether to adjust rewards by subtracting a fixed value (default is True).
    device : str | torch.device | None, optional
        The device on which tensors will be stored and computed (default is 'cpu').

    Attributes
    ----------
    gamma : float
        The discount factor for future rewards.
    gae_lambda : float
        The GAE lambda value for advantage calculation.
    adjust_rewards : bool
        Flag to indicate if rewards should be adjusted.
    device : torch.device
        The device tensors are stored on.
    memory : list
        A list of experience tuples collected during training.

    Methods
    -------
    push(state, action, log_prob, reward, done, value)
        Adds an experience tuple to the memory.
    create_batch(last_value)
        Processes the stored experiences into a batch for training.
    compute_returns_advantages(last_value, rewards, dones, values)
        Computes the returns and advantages using the GAE algorithm.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        adjust_rewards: bool = True,
        device: str | torch.device | None = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.adjust_rewards = adjust_rewards
        self.device = device
        self.memory = []

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
    ) -> None:
        """
        Stores an experience tuple in the memory.

        Parameters
        ----------
        state : torch.Tensor
            The state observed from the environment.
        action : torch.Tensor
            The action taken in the environment.
        log_prob : torch.Tensor
            The log probability of taking the action given the state.
        reward : float
            The reward received after taking the action.
        done : bool
            A boolean flag indicating whether the episode has ended.
        value : torch.Tensor
            The value estimate of the state.
        """

        # increase penalties for longer episodes
        if self.adjust_rewards:
            reward = reward - 0.3

        # add experience
        self.memory.append((state, action, log_prob, reward, done, value))

    def create_batch(
        self, last_value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes the memory to create a training batch.

        Parameters
        ----------
        last_value : torch.Tensor
            The value estimate for the final state.

        Returns
        -------
        states : torch.Tensor
            The states collected from the environment.
        actions : torch.Tensor
            The actions taken in the environment.
        old_log_probs : torch.Tensor
            The log probabilities of taking the actions given the states.
        returns : torch.Tensor
            The returns for the agent according to GAE.
        advantages : torch.Tensor
            The advantages for the agent according to GAE.
        """

        # unpack experience
        states, actions, log_probs, rewards, dones, values = zip(*self.memory)

        # format data
        old_log_probs = torch.cat(log_probs)
        states = torch.stack(states)
        actions = torch.stack(actions).unsqueeze(1)

        # calculate returns and advantages according to GAE
        returns, advantages = self.compute_returns_advantages(
            last_value, rewards, dones, values
        )

        # clear memory experience
        self.memory = []

        return states, actions, old_log_probs, returns, advantages

    def compute_returns_advantages(
        self,
        last_value: torch.Tensor,
        rewards: tuple[float],
        dones: tuple[bool],
        values: tuple[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the returns and advantages using GAE.

        Parameters
        ----------
        last_value : torch.Tensor
            The value estimate for the final state.
        rewards : tuple[float]
            A tuple of rewards collected during the episode.
        dones : tuple[bool]
            A tuple of boolean flags indicating if each step is terminal.
        values : tuple[torch.Tensor]
            A tuple of value estimates at each step.

        Returns
        -------
        returns : torch.Tensor
            The returns for the agent according to GAE.
        advantages : torch.Tensor
            The advantages for the agent according to GAE.
        """

        with torch.no_grad():

            # convert data to tensors
            rewards = torch.tensor(rewards, device=self.device)
            dones = torch.tensor(dones, device=self.device)
            values = torch.tensor(values, device=self.device)

            # initialize returns and advantages
            returns = torch.zeros(rewards.shape, device=self.device)
            advantages = torch.zeros(rewards.shape, device=self.device)

            # compute the returns and advantages using GAE
            gae = 0
            next_value = last_value
            for i in reversed(range(len(rewards))):
                if dones[i]:
                    next_value = 0
                delta = rewards[i] + self.gamma * next_value - values[i]
                gae = delta + self.gamma * self.gae_lambda * gae
                next_value = values[i]
                returns[i] = gae + values[i]
                advantages[i] = gae

            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages


class PPOAgent:
    """
    A Proximal Policy Optimization (PPO) agent, implementing a policy gradient method
    for reinforcement learning.

    Parameters
    ----------
    state_dim : int | tuple[int]
        The dimensionality of the state inputs.
    action_dim : int
        The number of discrete actions the agent can take.
    policy_lr : float, optional
        The learning rate for the policy optimizer (default is 4e-4).
    value_lr : float, optional
        The learning rate for the value optimizer (default is 2e-3).
    opt_eps : float, optional
        The epsilon term for numerical stability in the RMSprop optimizer
        (default is 1e-6).
    clip_eps : float, optional
        The clipping range epsilon, used to limit the ratio of new to old
        policy probabilities (default is 0.2).
    clip_grad : float, optional
        The gradient clipping value to prevent excessively large gradients during
        backpropagation (default is 1).
    update_epochs : int, optional
        The number of epochs to use for each update cycle (default is 10).
    entropy_coef : float, optional
        Coefficient for entropy bonus added to the loss function (default is 0.001).
    device : str | torch.device | None, optional
        The device on which tensors will be processed (default is 'cpu').

    Attributes
    ----------
    policy_net : DenseNet
        The neural network model that determines the policy.
    value_net : DenseNet
        The neural network model that estimates state values.
    policy_optimizer : optim.RMSprop
        The optimizer for updating the policy network.
    value_optimizer : optim.RMSprop
        The optimizer for updating the value network.
    MseLoss : nn.MSELoss
        The mean squared error loss function used for value estimation.
    """

    def __init__(
        self,
        state_dim: int | tuple[int],
        action_dim: int,
        policy_lr: float = 4e-4,
        value_lr: float = 2e-3,
        opt_eps: float = 1e-6,
        clip_eps: float = 0.2,
        clip_grad: float = 1,
        update_epochs: int = 10,
        entropy_coef: float = 0.001,
        device: str | torch.device | None = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # initialize policy and value networks and optimizers
        self.policy_net = DenseNet(self.state_dim, self.action_dim).to(self.device)
        self.value_net = DenseNet(self.state_dim).to(self.device)
        self.policy_optimizer = optim.RMSprop(
            self.policy_net.parameters(), lr=policy_lr, eps=opt_eps
        )
        self.value_optimizer = optim.RMSprop(
            self.value_net.parameters(), lr=value_lr, eps=opt_eps
        )

        self.clip_eps = clip_eps
        self.clip_grad = clip_grad
        self.update_epochs = update_epochs
        self.entropy_coef = entropy_coef

        # loss criterion
        self.MseLoss = nn.MSELoss()

    def select_action(
        self, state: np.ndarray | torch.Tensor, estimate_value: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Selects an action based on the current policy.

        Parameters
        ----------
        state : np.ndarray | torch.Tensor
            The current state from the environment.
        estimate_value : bool, optional
            Whether to estimate the value of the current state (default is True).

        Returns
        -------
        torch.Tensor
            The selected action as a tensor.
        torch.Tensor
            The log probability of the selected action.
        torch.Tensor | None
            The estimated value of the state or None if estimate_value is False.
        """

        with torch.no_grad():

            # convert to tensor
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, device=self.device)

            # compute the action and log probability according to the policy
            logits = self.policy_net(state.unsqueeze(0))
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # compute the state value according to the value network
            value = None
            if estimate_value:
                value = self.value_net(state)

        return action, log_prob, value

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> None:
        """
        Updates the policy and value networks based on provided training data.

        Parameters
        ----------
        states : torch.Tensor
            Tensor of all collected state observations.
        actions : torch.Tensor
            Tensor of all actions taken.
        old_log_probs : torch.Tensor
            Tensor of log probabilities of each action taken, as computed
            by the old policy.
        returns : torch.Tensor
            Tensor of calculated returns.
        advantages : torch.Tensor
            Tensor of calculated advantages.
        """

        # train/optimize the agent for update_epochs times
        for _ in range(self.update_epochs):

            # compute current logits, distribution and
            # log probabilities given the current policy
            logits = self.policy_net(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions.squeeze())

            # compute the ratio of the new log probabilities to the old ones
            ratios = torch.exp(new_log_probs - old_log_probs)

            # calculate the first surrogate loss (unclipped)
            surr1 = ratios * advantages

            # calculate the second surrogate loss (clipped)
            surr2 = (
                torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            )

            # compute the entropy loss to encourage exploration
            loss_entropy = self.entropy_coef * dist.entropy().mean()

            # compute the value loss using mean squared error
            loss_value = 0.5 * self.MseLoss(returns, self.value_net(states).squeeze())

            # compute the total policy loss
            loss_policy = -torch.min(surr1, surr2).mean() - loss_entropy

            # adjust weights and optimize according to loss and optimizer
            self.policy_optimizer.zero_grad()
            loss_policy.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_grad)
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.clip_grad)
            self.value_optimizer.step()


def train_model(
    env: Env,
    agent: PPOAgent,
    memory: Memory,
    max_episodes: int = 1000,
    max_steps: int = 700,
    smooth_window: int = 10,
) -> None:
    """
    Trains a PPO agent in a given environment using a specified memory buffer
    to store experiences.

    Parameters
    ----------
    env : Env
        The environment in which the agent is trained, which must adhere
        to the OpenAI Gym interface.
    agent : PPOAgent
        The PPO agent to be trained.
    memory : Memory
        The memory buffer used to store agent experiences.
    max_episodes : int, optional
        The maximum number of training episodes (default is 1000).
    max_steps : int, optional
        The maximum number of steps to execute per episode (default is 700).
    smooth_window : int, optional
        The number of episodes over which to smooth the displayed
        reward curve (default is 10).
    """

    episode_rewards = []
    for episode in range(max_episodes):
        # start new episode
        state = env.reset()[0]
        state = torch.tensor(state, device=agent.device)
        episode_reward = 0

        # play the episode
        for t in range(max_steps):
            # get the agent action, log probability and value of the state
            action, log_prob, value = agent.select_action(state)

            # execute the action in the environment
            next_state, reward, done, _, _ = env.step(action.item())

            # save the experience in the memory
            memory.push(state, action, log_prob, reward, done, value)

            # set new state
            state = torch.tensor(next_state, device=agent.device)
            episode_reward += reward

            # stop early if environment done
            if done:
                break

        # train the agent
        agent.update(*memory.create_batch(agent.value_net(state)))

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{max_episodes}, Reward: {episode_reward}")

    # plot the episode rewards during the training
    smoothed_rewards = np.convolve(
        episode_rewards, np.ones(smooth_window) / smooth_window, mode="valid"
    )
    plt.plot(smoothed_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Episode Rewards")
    plt.show()


def evaluate_model(
    env: Env, agent: PPOAgent, num_episodes: int = 10, max_steps: int = 800
) -> None:
    """
    Evaluates a trained PPO agent in a specified environment over a given number
    of episodes.

    Parameters
    ----------
    env : Env
        The environment in which the agent is evaluated, which must adhere
        to the OpenAI Gym interface.
    agent : PPOAgent
        The trained PPO agent to be evaluated.
    num_episodes : int, optional
        The number of episodes for evaluation (default is 10).
    max_steps : int, optional
        The maximum number of steps to execute per episode during evaluation
        (default is 800).
    """

    # play the evaluation episodes
    for episode in range(num_episodes):
        # start new episode
        state = env.reset()[0]
        episode_reward = 0

        for t in range(max_steps):
            # render the environment
            env.render()

            # get the agent action
            action, _, _ = agent.select_action(state, estimate_value=False)

            # execute the action in the environment
            state, reward, done, _, _ = env.step(action.item())

            episode_reward += reward
            if done:
                print(
                    f"Evaluation Episode {episode + 1}/{num_episodes} finished with reward: {episode_reward}"
                )
                break


def save_model(
    policy_net: nn.Module, value_net: nn.Module, filename: str = None
) -> None:
    """
    Saves the states of the agent's policy and value networks to a file.

    Parameters
    ----------
    policy_net : nn.Module
        The policy network to be saved.
    value_net : nn.Module
        The value network to be saved.
    filename : str, optional
        The path to the file where the network states should be saved.
    """

    if filename is None:
        filename = os.path.join(os.getcwd(), "models", "model.pt")

    torch.save(
        {
            "policy_net_state_dict": policy_net.state_dict(),
            "value_net_state_dict": value_net.state_dict(),
        },
        filename,
    )

    print(f"Model saved to {filename}")


def load_model(
    policy_net: nn.Module, value_net: nn.Module, filename: str = None
) -> None:
    """
    Loads the states of the agent's policy and value networks from a file.

    Parameters
    ----------
    policy_net : nn.Module
        The policy network into which the state should be loaded.
    value_net : nn.Module
        The value network into which the state should be loaded.
    filename : str, optional
        The path to the file from which the network states should be loaded.
    """

    if filename is None:
        filename = os.path.join(os.getcwd(), "models", "model.pt")

    checkpoint = torch.load(filename)
    policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    value_net.load_state_dict(checkpoint["value_net_state_dict"])

    print(f"Model loaded from {filename}")


def run() -> None:
    """
    Executes the full pipeline for training, saving, loading and evaluating
    a reinforcement learning agent on an OpenAI Gym environment.
    """

    # set device for the computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
    gamma = 0.99
    gae_lambda = 0.95

    policy_lr = 4e-4
    value_lr = 2e-3
    opt_eps = 1e-6
    clip_eps = 0.2
    clip_grad = 1
    update_epochs = 10
    entropy_coef = 0.001

    max_episodes = 1000
    max_steps = 700
    smooth_window = 10

    eval_num_episodes = 10
    eval_max_steps = 800

    # initialize the train environment
    train_env: Env = gym.make("LunarLander-v2", enable_wind=True, wind_power=5)

    # setup input and output dims
    state_dim: int = train_env.observation_space.shape[0]
    action_dim: int = train_env.action_space.n

    # initialize the agent
    agent = PPOAgent(
        state_dim,
        action_dim,
        policy_lr=policy_lr,
        value_lr=value_lr,
        opt_eps=opt_eps,
        clip_eps=clip_eps,
        clip_grad=clip_grad,
        update_epochs=update_epochs,
        entropy_coef=entropy_coef,
        device=device,
    )

    # initialize the agent memory
    memory = Memory(gamma=gamma, gae_lambda=gae_lambda, device=device)

    # train the model
    train_model(
        train_env,
        agent,
        memory,
        max_episodes=max_episodes,
        max_steps=max_steps,
        smooth_window=smooth_window,
    )

    train_env.close()

    # save weights
    save_model(agent.policy_net, agent.value_net)

    # initialize and load model independently of training
    agent = PPOAgent(
        state_dim,
        action_dim,
        policy_lr=policy_lr,
        value_lr=value_lr,
        opt_eps=opt_eps,
        clip_eps=clip_eps,
        clip_grad=clip_grad,
        update_epochs=update_epochs,
        entropy_coef=entropy_coef,
        device=device,
    )
    load_model(agent.policy_net, agent.value_net)

    # initialize the test environment
    eval_env: Env = gym.make("LunarLander-v2", render_mode="human")

    # evaluate the model
    evaluate_model(
        eval_env, agent, num_episodes=eval_num_episodes, max_steps=eval_max_steps
    )

    eval_env.close()


if __name__ == "__main__":
    run()
