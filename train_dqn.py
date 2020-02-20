# %%
import os
from rl_trading.envs.market_env import MarketEnv
import torch
import numpy as np
from sklearn import preprocessing
from rl_trading.agent.dqn import DQNAgent
import time
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--load', default=False, type=bool)  # load model
parser.add_argument('--epoches', default=30, type=int)  #  num of  games
parser.add_argument('--random_start', default='False', type=str)
parser.add_argument('--directory', default="./pytorch_models", type=str)
parser.add_argument('--episode_steps', default=50000, type=int)
parser.add_argument('--trading_period', default=5000, type=int)
parser.add_argument('--history_length', default=10, type=int)
parser.add_argument('--cash', default=300000, type=int)

args = parser.parse_args()

# start_timesteps = 5e4  # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
# eval_freq = 5e3  # How often the evaluation step is performed (after how many timesteps)
# max_timesteps = 1e6  # Total number of iterations/timesteps
# save_models = True  # Boolean checker whether or not to save the pre-trained model
# expl_noise = 0.1  # Exploration noise - STD value of exploration Gaussian noise
# eval_episodes = 1

env_name = 'cu_trading'
file_name = "%s_%s" % ("DQN", env_name)
random_start = True if args.random_start == 'True' else False

print("---------------------------------------")
print("Settings: %s" % (file_name))
print("---------------------------------------")

if not os.path.exists("./results"):
    os.makedirs("./results")
if not os.path.exists(args.directory):
    os.makedirs(args.directory)

kwargs = {
    'cash': args.cash,
    'unit': 5,
    'ratio': 0.07,
    'history_length': args.history_length,
    'start_step': 1000,
    'random_start': random_start,
    'trading_period': args.trading_period
}
env = MarketEnv(**kwargs)

seed = 0
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
num_actions = 3


def main():
    print("---------------------------------------")
    print("Collection Experience...")
    print("---------------------------------------")

    agent = DQNAgent(state_dim, num_actions, directory=args.directory)
    global_step = 0

    if args.load:
        agent.load(file_name)
    for epoch in range(args.epoches):
        state = env.reset()
        expect_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action, False)
            agent.memory.add(
                (state, next_state, action, reward, np.float(done)))
            expect_reward += reward

            if len(agent.memory.storage) >= 1000:
                agent.update(1)

            state = next_state
            if done:
                # agent.writer.add_scalar('expect_reward',
                #                         expect_reward,
                #                         global_step=global_step)
                print(
                    "Ep_i \t{}, the ep_r is \t{:0.2f}, trading period is \t{}, cash is \t{:0.2f}"
                    .format(epoch, expect_reward, env.current_step-env.start_step, env.current_cash))
                break
            global_step += 1

        agent.save(file_name)


if __name__ == '__main__':
    main()