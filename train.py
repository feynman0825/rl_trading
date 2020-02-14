# %%
import os
from rl_trading.envs.market_env import MarketEnv
import torch
import numpy as np
from sklearn import preprocessing
from rl_trading.agent.td3 import TD3
import time
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--epoches', default=30, type=int) #  num of  games
parser.add_argument('--policy_noise', default=0.2, type=float)
parser.add_argument('--noise_clip', default=0.5, type=float)
parser.add_argument('--exploration_noise', default=0.1, type=float)
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
file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
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
action_dim = env.action_space.shape[0]
max_action = 1


# evaluations = []
# total_timesteps = 0
# timesteps_since_eval = 0
# episode_num = 0
# done = False
# episode_reward = 0
# episode_timesteps = 0
# obs = env.reset()
# t0 = time.time()
# epoches = 10

# # We start the main loop over 500,000 timesteps
# # while total_timesteps < max_timesteps:
# for epoch in tqdm(range(epoches)):
#     while not done:
#         # Before 10000 timesteps, we play random actions
#         # If the explore_noise parameter is not 0, we add noise to the action and we clip it
#         if total_timesteps < start_timesteps:
#             action = env.action_space.sample()
#         else:  # After 10000 timesteps, we switch to the model
#             action = policy.select_action(np.array(obs))

#         # The agent performs the action in the environment, then reaches the next state and receives the reward
#         new_obs, reward, done, _ = env.step(action)

#         # We check if the episode is done
#         done_bool = 0 if episode_timesteps + 1 == env.total_period else float(done)

#         # We increase the total reward
#         episode_reward += reward

#         # We store the new transition into the Experience Replay memory (ReplayBuffer)
#         transition = (obs, new_obs, action, reward, done_bool)
#         policy.update(transition, episode_timesteps)

#         # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
#         obs = new_obs
#         episode_timesteps += 1
#         total_timesteps += 1
#         timesteps_since_eval += 1

#     print("Total Timesteps: {} Episode Num: {} Reward: {}".format(
#         total_timesteps, episode_num, episode_reward))
#     # We evaluate the episode and we save the policy
#     if timesteps_since_eval >= eval_freq:
#         timesteps_since_eval %= eval_freq
#         evaluations.append(
#             env.evaluate_policy(policy, eval_episodes=eval_episodes))
#         policy.save(file_name, directory="./pytorch_models")
#         policy.save_replay(file_name, directory="./pytorch_models")
#         np.save("./results/%s" % (file_name), evaluations)

#     # When the training step is done, we reset the state of the environment
#     obs = env.reset()

#     # Set the Done to False
#     done = False

#     # Set rewards and episode timesteps to zero
#     episode_reward = 0
#     episode_timesteps = 0
#     episode_num += 1

# We add the last policy evaluation to our list of evaluations and we save our model
# evaluations.append(env.evaluate_policy(agent, eval_episodes=eval_episodes))
# if save_models:
#     agent.save("%s" % (file_name))
#     agent.save_replay(file_name)
# np.save("./results/%s" % (file_name), evaluations)

def main():
    print("---------------------------------------")
    print("Collection Experience...")
    print("---------------------------------------")

    agent = TD3(state_dim, action_dim, max_action, directory=args.directory)
    global_step = 0

    if args.load:
        agent.load(file_name)
    for epoch in range(args.epoches):
        state = env.reset()
        episode_step = 0
        expect_reward = 0.0
        while episode_step < args.episode_steps:
            action = agent.select_action(state)
            action = action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])
            action = action.clip(env.action_space.low, env.action_space.high)[0]
            next_state, reward, done, _ = env.step(action)

            agent.memory.add((state, next_state, action, reward, np.float(done)))

            expect_reward += reward

            # if epoch+1 % 10 == 0:
            #     print('Episode {},  The memory size is {} '.format(epoch, len(agent.memory.storage)))
            if len(agent.memory.storage) >= agent.capacity-1:
                agent.update(10)

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