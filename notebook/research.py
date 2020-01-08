# %%
import os
from rl_trading.envs.market_env import MarketEnv
import torch
import numpy as np
from sklearn import preprocessing
from rl_trading.agent.td3 import TD3, ReplayBuffer
import time

seed = 0  # Random seed number
start_timesteps = 5e3  # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5e3  # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 6e5 # 5e5  # Total number of iterations/timesteps
save_models = True  # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1  # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100  # Size of the batch
discount = 0.99  # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005  # Target network update rate
policy_noise = 0.2  # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5  # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2  # Number of iterat
eval_episodes = 10

env_name = 'cu_trading'
file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
print("---------------------------------------")
print("Settings: %s" % (file_name))
print("---------------------------------------")

if not os.path.exists("./results"):
    os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

kwargs = {'cash': 1e5, 'unit': 5, 'ratio': 0.07, 'target_profit_factor': 10}
env = MarketEnv(**kwargs)

env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = env.observation_space.shape[0]
action_dim = len(env.action_space)
# env._data = env._data.head(100)

# %%
env._data.shape
# %%

policy = TD3(state_dim, action_dim)
replay_buffer = ReplayBuffer()

def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print("---------------------------------------")
    print("Average Reward over the Evaluation Step: %f" % (avg_reward))
    print("---------------------------------------")
    return avg_reward

evaluations = [evaluate_policy(policy, 10)]
total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
t0 = time.time()

# We start the main loop over 500,000 timesteps
while total_timesteps < max_timesteps:
  
    # If the episode is done
    if done:

        # If we are not at the very beginning, we start the training process of the model
        if total_timesteps != 0:
            print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
            policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

        # We evaluate the episode and we save the policy
        if timesteps_since_eval >= eval_freq:
            timesteps_since_eval %= eval_freq
            evaluations.append(evaluate_policy(policy, eval_episodes=eval_episodes))
            policy.save(file_name, directory="./pytorch_models")
            np.save("./results/%s" % (file_name), evaluations)

        # When the training step is done, we reset the state of the environment
        obs = env.reset()

        # Set the Done to False
        done = False

        # Set rewards and episode timesteps to zero
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # Before 10000 timesteps, we play random actions
    if total_timesteps < start_timesteps:
        action = float(np.random.choice(env.action_space))#.sample()
    else: # After 10000 timesteps, we switch to the model
        action = policy.select_action(np.array(obs))
    # If the explore_noise parameter is not 0, we add noise to the action and we clip it
    # if expl_noise != 0:
    #     action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

    # The agent performs the action in the environment, then reaches the next state and receives the reward
    new_obs, reward, done, _ = env.step(action)

    # We check if the episode is done
    done_bool = 0 if episode_timesteps + 1 == env.total_period else float(done)

    # We increase the total reward
    episode_reward += reward

    # We store the new transition into the Experience Replay memory (ReplayBuffer)
    replay_buffer.add((obs, new_obs, action, reward, done_bool))

    # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
    obs = new_obs
    episode_timesteps += 1
    total_timesteps += 1
    timesteps_since_eval += 1

# We add the last policy evaluation to our list of evaluations and we save our model
evaluations.append(evaluate_policy(policy, eval_episodes=eval_episodes))
if save_models:
    policy.save("%s" % (file_name), directory="./pytorch_models")
np.save("./results/%s" % (file_name), evaluations)
