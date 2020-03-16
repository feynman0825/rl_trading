"""Market environment
This is the rl trading envirnment for rl-agent to trading and simulate.
"""

# %%
import logging
import os
import numpy as np
import pandas as pd
import gym
from rl_trading.utils.constant import *
from gym import spaces
from sklearn import preprocessing
import pandas_ta as ta

logger = logging.getLogger(__name__)

DTYPE = {
    'open': 'int32',
    'close': 'int32',
    'high': 'int32',
    'low': 'int32',
    'volume': 'int32',
    'open_interest': 'int32'
}

AGG_PARAMS = {
    'high': 'max',
    'low': 'min',
    'open': 'first',
    'close': 'last',
    'volume': 'sum',
    'open_interest': 'sum'
}


class MarketEnv(gym.Env):
    ''' Market environment
    '''
    def __init__(self, *args, **kwargs):
        super(MarketEnv).__init__()
        self.initial_cash = kwargs['cash']
        self.unit = kwargs['unit']
        self.ratio = kwargs['ratio']
        self.history_length = kwargs['history_length']
        self.start_step = kwargs['start_step']
        self.trading_period = kwargs['trading_period']
        self.random_start = kwargs['random_start']
        self.transaction_cost = 15
        self.sigma_target = 1
        self.load_data()
        self.extract_features()
        self.action_space = spaces.Box(
            -1, 1, shape=(1, ))  #np.array([FLAT, LONG, SHORT])
        self.observation_space = spaces.Box(0,
                                            1,
                                            shape=(self.features.shape[1],
                                                   self.history_length))
        # self._transformer = FeatureTransformer(self.data)
        # self.transformed_obs = self._transformer.transform(self.data)
        self.reset()

    def reset(self):
        """Reset the state of the environment and returns an initial observation.
        Returns:
            numpy.array: The initial observation of the space. Initial reward is assumed to be 0.
        """
        # self.current_position = FLAT
        self.second_last_action = FLAT
        self.total_pnl = 0.0
        self.current_cash = self.initial_cash
        if self.random_start:
            self.current_step = np.random.randint(self.start_step, 200000)
        else:
            self.current_step = self.start_step
        self.total_pnl = 0
        self.current_price = self.data.iloc[self.current_step, 3]
        self.last_price = self.data.iloc[self.current_step - 1, 3]
        self.last_margin = 0
        return self.get_observation()

    def load_data(self, freq='5T'):
        path = 'data/CU9999.csv'
        cu_df = pd.read_csv(path,
                            infer_datetime_format=True,
                            parse_dates=[1],
                            dtype=DTYPE)
        cu_df.drop(['Unnamed: 0', 'money'], axis=1, inplace=True)
        cu_df = cu_df.set_index('date').resample(freq).agg(AGG_PARAMS).dropna()
        cu_df = cu_df.astype(DTYPE)
        self.data = cu_df

    @property
    def total_period(self):
        return self.data.shape[0]

    def step(self, action, continous=True):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (int): FLAT, LONG, SHORT
        Returns:
            tuple:
                - observation (numpy.array): Agent's observation of the current environment.
                - reward (float) : Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended, in which case
                               further step() calls will return undefined results.
                - info (str): Contains auxiliary diagnostic information
                              (helpful for debugging, and sometimes learning).
        """
        logger.info('Agent at time step of %d takes %s action' %
                    (self.current_step, action))
        if continous:
            action = LONG if action > 0.5 else FLAT if action > -0.5 else SHORT
        else:
            action -= 1

        # main logic
        pnl = 0.0
        reward = 0

        diff_price = self.current_price - self.last_price
        pnl = self.unit * diff_price * action - self.transaction_cost * abs(
            self.second_last_action - action) * abs(action)  # 平仓不收取手续费

        # just for now
        self.current_cash += pnl

        # adjust for margin
        if self.second_last_action == FLAT and action in (LONG, SHORT):
            # open new position
            self.last_margin = self.unit * self.last_price * self.ratio
            self.current_cash -= self.last_margin
        elif action == FLAT and self.second_last_action in (LONG, SHORT):
            # close position
            self.current_cash += self.last_margin
            self.last_margin = 0
        elif (action == LONG and self.second_last_action == SHORT) or (
                action == SHORT and self.second_last_action == LONG):
            # close position
            self.current_cash += self.last_margin
            # then open new position
            self.last_margin = self.unit * self.last_price * self.ratio
            self.current_cash -= self.last_margin

        self.total_pnl += pnl

        # update reward
        reward = self.get_reward(action)

        # increment time step
        self.current_step += 1

        # update price track
        self.last_price = self.current_price
        self.current_price = self.data.iloc[self.current_step, 3]

        # update position
        self.second_last_action = action

        observation = self.get_observation()
        done = self.is_trading_done()
        info = {}  # modify later

        return observation, reward, done, info

    def get_reward(self, action) -> float:
        # mu = 1 #  a fixed number per contract at each trade
        #reward = mu * sigma_target * (action/sigma*r - bp * price * abs(action/sigma-prev_action/prev_sigma))
        # sigma include previous data
        # window = min(250, self.start_step)
        # sigma = (
        #     self.data.close.iloc[self.current_step + 1 -
        #                          window:self.current_step + 1].to_numpy() -
        #     self.data.close.iloc[self.current_step -
        #                          window:self.current_step].to_numpy()).std()
        diff_price = self.current_price - self.last_price
        # reward = self.sigma_target / sigma * (
        #     self.unit * action * diff_price - self.transaction_cost *
        #     abs(self.second_last_action - action) * abs(action))

        reward = (self.unit * action * diff_price - self.transaction_cost *
                  abs(self.second_last_action - action) * abs(action)) / (
                      self.last_price * self.unit) / self.ratio * 100

        # reward clip
        reward = np.clip(reward, -2, 2)
        return reward

    def is_trading_done(self) -> bool:
        if self.trading_period is not None:
            if self.current_step >= self.start_step + self.trading_period:
                return True
        # check whether account goes broke
        minimal_margin = self.current_price * self.unit * self.ratio
        if self.current_cash <= minimal_margin:
            return True
        # check whether simulation is done
        if self.current_step >= self.total_period - 1:
            return True
        return False

    def get_observation(self) -> np.array:
        """Get the observation matrix
        There are two components of the matrix, the asset_matrix and position_matrix,
        Shape of asset_matrix is (time_step,features), and shape of position_matrix is
        (time_step, 1).
        Returns:
            numpy.array: Return the concatenated matrix, which is (time_step, features+1)
        """
        # transformed_obs = self._transformer.transform(self.data.iloc[self.current_step-self.history_length:self.current_step])

        # observation = (
        #     self.transformed_obs[self.current_step -
        #                          self.history_length:self.current_step, :],
        #     self.second_last_action)

        # normalized close

        observation = self.features[self.current_step -
                                    self.history_length:self.current_step, :]
        return observation

    def _render(self, mode='human'):
        """Render the environment
        """
        pass

    def extract_features(self):
        min_max_scaler = preprocessing.MinMaxScaler()

        # RSI
        # ref in https://arxiv.org/pdf/1911.10107.pdf
        rsi = ta.rsi(self.data.close, length=28)
        normalized_rsi = rsi.to_numpy().reshape(-1,
                                                1) / 100 - 0.5  # (-0.5, 0.5)

        # normalized close
        # ref in https://arxiv.org/pdf/1911.10107.pdf
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_close = min_max_scaler.fit_transform(
            self.data.close.to_frame()).reshape(-1, 1) - 0.5

        # normalized return
        # ref in https://arxiv.org/pdf/1911.10107.pdf
        ret_window = 60
        ret_60 = self.data.close / self.data.close.shift(ret_window) - 1
        normalized_ret_60 = ret_60 / ta.stdev(ret_60, ret_window)#ret_60.rolling(ret_window).std()
        normalized_ret_60 = normalized_ret_60.to_numpy().reshape(-1, 1)

        ret_window = 120
        ret_120 = self.data.close / self.data.close.shift(ret_window) - 1
        normalized_ret_120 = ret_120 / ta.stdev(ret_120, ret_window)
        normalized_ret_120 = normalized_ret_120.to_numpy().reshape(-1, 1)

        # MACD indicators
        # ref in https://arxiv.org/pdf/1911.10107.pdf
        qt = (ta.sma(self.data.close, 30) -
              ta.sma(self.data.close, 60)) / ta.stdev(self.data.close, 60)
        macd = qt / ta.stdev(qt, 60)
        macd = macd.to_numpy().reshape(-1, 1)

        # concat features
        features = np.concatenate(
            (normalized_rsi, normalized_close, normalized_ret_60,
             normalized_ret_120, macd),
            axis=1)
        self.features = features

    def evaluate_policy(self, policy, eval_episodes=10, max_steps=None):
        avg_reward = 0.
        for _ in range(eval_episodes):
            obs = self.reset()
            done = False
            while not done:
                action = policy.select_action(np.array(obs))
                obs, reward, done, _ = self.step(action)
                avg_reward += reward
                if self.current_step % 1e4 == 0:
                    print('Evaluate policy: ', self.current_step, reward, done,
                        self.current_cash, action)
                if max_steps is not None and self.current_step > max_steps:
                    break
        avg_reward /= eval_episodes
        print("---------------------------------------")
        print("Average Reward over the Evaluation Step: %f" % (avg_reward))
        print("---------------------------------------")
        return avg_reward

# class FeatureTransformer:
#     def __init__(self, X):
#         self.scaler = preprocessing.StandardScaler()#.MinMaxScaler()
#         self.scaler.fit(X)

#     def transform(self, data):
#         return self.scaler.transform(data)
