import torch
import torch.nn as nn
from torch.distributions import Categorical
import threading
import os.path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def count(self):
        return len(self.rewards)

    def pop(self):
        return self.actions.pop(), self.states.pop(), self.logprobs.pop(), self.rewards.pop(), self.is_terminals.pop()

    def append(self, state, action, logprobs, reward, done):
        self.actions.append(action)
        self.states.append(state)
        self.logprobs.append(logprobs)
        self.rewards.append(reward)
        self.is_terminals.append(done)

    def extend(self, states, actions, logprobs, rewards, dones):
        self.actions.extend(actions)
        self.states.extend(states)
        self.logprobs.extend(logprobs)
        self.rewards.extend(rewards)
        self.is_terminals.extend(dones)


class ActorCritic(nn.Module):
    def __init__(self, layer_dims):
        super(ActorCritic, self).__init__()

        self._lock = threading.Lock()
        len_layer_dims = len(layer_dims)
        assert(len_layer_dims >= 2)

        # actor
        layers = []
        for i in range(len_layer_dims-2):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(layer_dims[len_layer_dims-2], layer_dims[len_layer_dims-1]))
        layers.append(nn.Softmax(dim=-1))
        self.action_layer = nn.Sequential(*layers)

        # critic
        layers = []
        for i in range(len_layer_dims-2):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(layer_dims[len_layer_dims-2], 1))
        self.value_layer = nn.Sequential(*layers)

        self.initialize_weights()

    @staticmethod
    def _init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def initialize_weights(self):
        self.action_layer.apply(fn=self._init_weights)
        self.value_layer.apply(fn=self._init_weights)

    def forward(self):
        raise NotImplementedError
        
    def act(self, state):
        with self._lock:
            state_tensor = torch.from_numpy(state).float().to(device)
            action_probs = self.action_layer(state_tensor)
            # dist = Categorical(action_probs)
            # action = dist.sample()

            return state_tensor, action_probs         # state, dist, action, dist.log_prob(action)

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, layer_dims, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # self.policy_static = ActorCritic(layer_dims).to(device)

        self.policy = ActorCritic(layer_dims).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(layer_dims).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def load_policy_from_file(self, saved_file):
        if os.path.isfile(saved_file):
            self.policy.load_state_dict(torch.load(saved_file))
            # self.policy_static.load_state_dict(torch.load(saved_file))
            # self.policy.load_state_dict(self.policy_static.state_dict())
        else:
            self.policy.initialize_weights()
        self.policy_old.load_state_dict(self.policy.state_dict())

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
