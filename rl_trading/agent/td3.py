import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from rl_trading.utils.hyperparameters import Config
from rl_trading.utils.replay_memory import ReplayBuffer
from tensorboardX import SummaryWriter
import pickle
import os

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action
        # initialize weight
        # nn.init.xavier_uniform_(self.layer_1.weight)
        # nn.init.xavier_uniform_(self.layer_2.weight)
        # nn.init.xavier_uniform_(self.layer_3.weight)

    def forward(self, x):
        # x = torch.flatten(x)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        # Defining the second Critic neural network
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, 1)
        # initialize weight
        # nn.init.xavier_uniform_(self.layer_1.weight)
        # nn.init.xavier_uniform_(self.layer_2.weight)
        # nn.init.xavier_uniform_(self.layer_3.weight)
        # nn.init.xavier_uniform_(self.layer_4.weight)
        # nn.init.xavier_uniform_(self.layer_5.weight)
        # nn.init.xavier_uniform_(self.layer_6.weight)

    def forward(self, x, u):
        # x = torch.flatten(x)
        xu = torch.cat([x, u], 1)
        # Forward-Propagation on the first Critic Neural Network
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Forward-Propagation on the second Critic Neural Network
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1



# Building the whole Training Process into a class

class TD3(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action=1,
                 directory="./pytorch_models"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.tau = Config.tau
        self.batch_size = Config.batch_size
        self.gamma = Config.gamma
        self.policy_noise = Config.policy_noise
        self.noise_clip = Config.noise_clip
        self.policy_freq = Config.policy_freq
        self.capacity = Config.capacity
        self.learning_rate = Config.learning_rate
        self.actor_losses = []
        self.critic_losses = []
        self.rewards = []
        self.directory = directory
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0
        # Selecting the device (CPU or GPU)
        self.device = Config.device
        self.declare_networks()
        self.declare_memory()


    def declare_memory(self):
        self.memory = ReplayBuffer(self.capacity)

    def declare_networks(self):
        self.actor = Actor(self.state_dim, self.action_dim,
                           self.max_action).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim,
                                  self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate)
        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.Tensor(state.reshape(1, -1)).to(self.device)
            return self.actor(state).cpu().data.numpy().flatten()

    def prep_minibatch(self):
        # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
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
        if self.num_training % 500 == 0:
            print("====================================")
            print("model has been trained for {} times...".format(self.num_training))
            print("====================================")

        for it in range(num_iteration):
            # get batch sample
            state, action, reward, next_state, done = self.prep_minibatch(
            )

            # Step 5: From the next state s’, the Actor target plays the next action a’
            next_action = self.actor_target(next_state)

            # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
            noise = torch.Tensor(action).data.normal_(
                0, self.policy_noise).to(self.device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action,
                                                      self.max_action)
            # new_obs, reward, done, _ = env.step(action)

            # # append to memory buffer
            # self.memory.add(transition)

           
            critic_loss = self.compute_critic_loss(state, action, next_action,
                                                reward, next_state, done)

            # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            for param in self.critic.parameters():
                param.grad.data.clamp_(-1, 1)
            self.critic_optimizer.step()
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)

            # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
            actor_loss = self.compute_actor_loss(state)
            if it % self.policy_freq == 0:
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                for param in self.critic.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.actor_optimizer.step()

                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

                # update target model
                self.update_target_model()
                self.num_actor_update_iteration += 1

        self.num_critic_update_iteration += 1
        self.num_training += 1

            # # save loss
            # actor_loss_eval = actor_loss.cpu().data.numpy().tolist()
            # critic_loss_eval = critic_loss.cpu().data.numpy().tolist()
            # self.save_loss(actor_loss_eval, critic_loss_eval)
            # self.save_reward(reward)
            

    def save_loss(self, actor_loss, critic_loss):
        self.critic_losses.append(critic_loss)
        self.actor_losses.append(actor_loss)

    def save_reward(self, reward):
        self.rewards.append(reward)


    def compute_critic_loss(self, state, action, next_action, reward,
                            next_state, done):
        gamma = self.gamma
        # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        #next_action.reshape(-1, 1))

        # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
        target_Q = torch.min(target_Q1, target_Q2)

        # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
        target_Q = reward + ((1 - done) * gamma * target_Q).detach()

        # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
        current_Q1, current_Q2 = self.critic(state, action.reshape(-1, 1))

        # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
        # critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
        #     current_Q2, target_Q)
        critic_loss = self.huber_loss(current_Q1, target_Q) + self.huber_loss(
            current_Q2, target_Q)
        return critic_loss

    def compute_actor_loss(self, state):
        return -self.critic.Q1(state, self.actor(state)).mean()

    def update_target_model(self):
        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.critic.parameters(),
                                       self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data +
                                    (1 - self.tau) * target_param.data)

        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.actor.parameters(),
                                       self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data +
                                    (1 - self.tau) * target_param.data)

    def huber_loss(self, x, y):
        diff = x - y
        cond = (diff.abs() < 1.0).float().detach()
        losses = 0.5 * diff.pow(2) * cond + (diff.abs() - 0.5) * (1.0 - cond)
        loss = losses.mean()
        return loss

    # Making a save method to save a trained model
    def save(self, filename):
        torch.save(self.actor.state_dict(),
                   '%s/%s_actor.pth' % (self.directory, filename))
        torch.save(self.critic.state_dict(),
                   '%s/%s_critic.pth' % (self.directory, filename))
        torch.save(self.actor.state_dict(),
                   '%s/%s_actor_target.pth' % (self.directory, filename))
        torch.save(self.critic.state_dict(),
                   '%s/%s_critic_target.pth' % (self.directory, filename))
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    # Making a load method to load a pre-trained model
    def load(self, filename):
        self.actor.load_state_dict(
            torch.load('%s/%s_actor.pth' % (self.directory, filename)))
        self.critic.load_state_dict(
            torch.load('%s/%s_critic.pth' % (self.directory, filename)))
        self.actor_target.load_state_dict(
            torch.load('%s/%s_actor_target.pth' % (self.directory, filename)))
        self.critic_target.load_state_dict(
            torch.load('%s/%s_critic_target.pth' % (self.directory, filename)))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

    def save_replay(self, filename):
        pickle.dump(self.memory, open('%s/%s_memroy.dump' % (self.directory, filename), 'wb'))

    def load_replay(self, filename):
        fname = '%s/%s_memroy.dump' % (self.directory, filename)
        if os.path.isfile(fname):
            self.memory = pickle.load(open(fname, 'rb'))
