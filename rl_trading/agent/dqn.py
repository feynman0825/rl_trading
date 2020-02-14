import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from rl_trading.utils.hyperparameters import Config
from rl_trading.utils.replay_memory import ReplayBuffer
import pickle
import os


class DQN(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, num_actions)
        self.num_actions = num_actions

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x


# Building the whole Training Process into a class


class DQNAgent:
    def __init__(self, state_dim, num_actions=3, directory="./pytorch_models"):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.tau = Config.tau
        self.batch_size = Config.batch_size
        self.gamma = Config.gamma
        self.capacity = Config.capacity
        self.learning_rate = Config.learning_rate
        self.target_net_update_freq = Config.target_net_update_freq
        self.static_policy = False
        self.directory = directory
        self.update_count = 0
        self.losses = []
        self.rewards = []
        # Selecting the device (CPU or GPU)
        self.device = Config.device
        self.declare_networks()
        self.declare_memory()

    def declare_memory(self):
        self.memory = ReplayBuffer(self.capacity)

    def declare_networks(self):
        self.model = DQN(self.state_dim, self.num_actions).to(self.device)
        self.target_model = DQN(self.state_dim,
                                self.num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)

    def select_action(self, state, eps=0.1):
        with torch.no_grad():
            if np.random.random() >= eps or self.static_policy:
                # X = torch.Tensor(state.reshape(1, -1),
                #                  device=self.device,
                #                  dtype=torch.float)
                state = torch.Tensor(state.reshape(1, -1)).to(self.device)
                a = self.model(state).max(1)[1].view(1, 1)
                #a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def prep_minibatch(self):
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = self.memory.sample(
            self.batch_size)
        batch_states = batch_states.reshape((self.batch_size, -1))
        batch_next_states = batch_next_states.reshape((self.batch_size, -1))
        state = torch.Tensor(batch_states).to(self.device)
        next_state = torch.Tensor(batch_next_states).to(self.device)
        action = torch.Tensor(batch_actions).to(self.device)
        reward = torch.Tensor(batch_rewards).to(self.device)
        done = torch.Tensor(batch_dones).to(self.device)

        return state, action, reward, next_state, done

    def update(self, num_iteration):
        # if self.num_training % 500 == 0:
        #     print("====================================")
        #     print("model has been trained for {} times...".format(self.num_training))
        #     print("====================================")

        for it in range(num_iteration):
            # get batch sample
            state, action, reward, next_state, done = self.prep_minibatch()

            next_action = self.target_model(next_state)

            loss = self.compute_loss(state, action, next_action, reward,
                                     next_state, done)

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            self.update_target_model()
            self.save_loss(loss.cpu().data.numpy().tolist())
            self.save_reward(reward)

        #if self.update_count % 10 == 0:
        moving_loss = np.mean(self.losses[-100:])
        print(
            f'At {self.update_count}, loss is: {moving_loss}'
        )

    def compute_loss(self, state, action, next_action, reward, next_state, done):
        gamma = self.gamma
        action = action.long().reshape([-1, 1])
        next_action = next_action.argmax(dim=1).reshape([-1, 1])
        
        current_Q = self.model(state).gather(1, action)
        with torch.no_grad():
            max_next_q_values = self.target_model(next_state).gather(1, next_action)

        target_Q = reward + ((1 - done) * gamma * max_next_q_values).detach()

        loss = self.huber_loss(current_Q, target_Q)
        return loss

    def save_loss(self, loss):
        self.losses.append(loss)

    def save_reward(self, reward):
        self.rewards.append(reward)

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def huber_loss(self, x, y):
        diff = x - y
        cond = (diff.abs() < 1.0).float().detach()
        losses = 0.5 * diff.pow(2) * cond + (diff.abs() - 0.5) * (1.0 - cond)
        loss = losses.mean()
        return loss

    # Making a save method to save a trained model
    def save(self, filename):
        torch.save(self.model.state_dict(),
                   '%s/%s_dqn.pth' % (self.directory, filename))
        torch.save(self.model.state_dict(),
                   '%s/%s_dqn_target.pth' % (self.directory, filename))

    # Making a load method to load a pre-trained model
    def load(self, filename):
        self.model.load_state_dict(
            torch.load('%s/%s_dqn.pth' % (self.directory, filename)))
        self.model.load_state_dict(
            torch.load('%s/%s_dqn_target.pth' % (self.directory, filename)))

    def save_replay(self, filename):
        pickle.dump(self.memory,
                    open('%s/%s_memroy.dump' % (self.directory, filename), 'wb'))

    def load_replay(self, filename):
        fname = '%s/%s_memroy.dump' % (self.directory, filename)
        if os.path.isfile(fname):
            self.memory = pickle.load(open(fname, 'rb'))
