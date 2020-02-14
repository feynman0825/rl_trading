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
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--epoches', default=30, type=int) #  num of  games
parser.add_argument('--policy_noise', default=0.2, type=float)
parser.add_argument('--noise_clip', default=0.5, type=float)
parser.add_argument('--directory', default="./pytorch_models", type=str)
parser.add_argument('--print_log', default=100, type=int)
parser.add_argument('--episode_steps', default=50000, type=int)
args = parser.parse_args()


seed = 0  # Random seed number
# start_timesteps = 5e4  # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
# eval_freq = 5e3  # How often the evaluation step is performed (after how many timesteps)
# max_timesteps = 1e6  # Total number of iterations/timesteps
# save_models = True  # Boolean checker whether or not to save the pre-trained model
# expl_noise = 0.1  # Exploration noise - STD value of exploration Gaussian noise
# eval_episodes = 1



env_name = 'cu_trading'
file_name = "%s_%s_%s" % ("DQN", env_name, str(seed))
print("---------------------------------------")
print("Settings: %s" % (file_name))
print("---------------------------------------")

if not os.path.exists("./results"):
    os.makedirs("./results")
if not os.path.exists(args.directory):
    os.makedirs(args.directory)

kwargs = {
    'cash': 5e5,
    'unit': 5,
    'ratio': 0.07,
    'history_length': 10,
    'start_step': 1000
}
env = MarketEnv(**kwargs)

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
        episode_step = 0
        expect_reward = 0.0
        while episode_step < args.episode_steps:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action, False)
            agent.memory.add((state, next_state, action, reward, np.float(done)))
            expect_reward += reward


            # if epoch+1 % 10 == 0:
            #     print('Episode {},  The memory size is {} '.format(epoch, len(agent.memory.storage)))
            if len(agent.memory.storage) >= 1000:#agent.capacity-1:
                agent.update(100)

            state = next_state
            if done:
                agent.writer.add_scalar('expect_reward', expect_reward, global_step=global_step)
                if epoch % args.print_log == 0:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, current eposode step is \t{}".format(epoch, expect_reward, episode_step))
                expect_reward = 0
            global_step += 1
            episode_step += 1

        agent.save(file_name)

if __name__ == '__main__':
    main()