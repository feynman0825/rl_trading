import torch
import math


class Config(object):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 1e-4

    # TD3 controls
    gamma = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
    tau = 0.005 # Target network update rate
    policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
    noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
    policy_freq = 2 # Number of iterat

    # memory
    batch_size = 100 # Size of the batch
    capacity = 1e6

    # DQN
    target_net_update_freq = 1000

    # #PPO controls
    # ppo_epoch = 3
    # num_mini_batch = 32
    # ppo_clip_param = 0.1

    # #a2c controls
    # num_agents = 8
    # rollout = 16
    # value_loss_weight = 0.5
    # entropy_loss_weight = 0.001
    # grad_norm_max = 0.5
    # USE_GAE = True
    # gae_tau = 0.95

    # #algorithm control
    # USE_NOISY_NETS = False
    # USE_PRIORITY_REPLAY = False
    # #Multi-step returns
    # N_STEPS = 1

    # #epsilon variables
    # epsilon_start = 1.0
    # epsilon_final = 0.01
    # epsilon_decay = 30000
    # epsilon_by_frame = lambda frame_idx: self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1. * frame_idx / self.epsilon_decay)

    # #misc agent variables
    # GAMMA = 0.99
    # LR = 1e-4

    # #memory
    # TARGET_NET_UPDATE_FREQ = 1000
    # EXP_REPLAY_SIZE = 100000
    # BATCH_SIZE = 32
    # PRIORITY_ALPHA = 0.6
    # PRIORITY_BETA_START = 0.4
    # PRIORITY_BETA_FRAMES = 100000

    # #Noisy Nets
    # SIGMA_INIT = 0.5

    # #Learning control variables
    # LEARN_START = 10000
    # MAX_FRAMES = 100000

    # #Categorical Params
    # ATOMS = 51
    # V_MAX = 10
    # V_MIN = -10

    # #Quantile Regression Parameters
    # QUANTILES = 51

    # #DRQN Parameters
    # SEQUENCE_LENGTH = 8

