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
        self._target_profit_factor = kwargs['target_profit_factor']
        self.history_length = kwargs['history_length']
        self.start_step = kwargs['start_step']
        self.action_space = np.array([FLAT, LONG, SHORT])
        self.observation_space = spaces.Box(0, 1, shape=(7,))
        self.transaction_cost = 15
        self.sigma_target = 1
        self._load_data()
        self._transformer = FeatureTransformer(self.data)
        self.transformed_obs = self._transformer.transform(self.data)
        self.reset()

    def reset(self):
        """Reset the state of the environment and returns an initial observation.
        Returns:
            numpy.array: The initial observation of the space. Initial reward is assumed to be 0.
        """
        self.current_position = FLAT
        self.total_pnl = 0.0
        self.current_cash = self.initial_cash
        self.current_step = max(0, self.start_step)
        self.total_pnl = 0
        self.current_price = self.data.iloc[self.current_step, 3]
        self.last_price = self.data.iloc[self.current_step-1, 3]
        self.current_position_pnl = 0
        self.last_margin = 0
        return self.get_observation()

    def _load_data(self, freq='5T'):
        path = 'data/cu9999.csv'
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


    def step(self, action):
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
        assert action in self.action_space, "Action should be either 0,-1 or 1"
        logger.info('Agent at time step of %d takes %s action' %
                    (self.current_step, action))

        # main logic
        current_position = self.current_position
        pnl = 0.0
        reward = 0

        diff_price = self.current_price - self.last_price
        pnl = self.unit * diff_price * current_position
        self.current_position_pnl += pnl

        if self.current_position == FLAT and action in (LONG, SHORT):
            # open new position
            pnl -= self.transaction_cost
            self.current_position_pnl -= self.transaction_cost
            self.last_margin = self.unit * self.current_price * self.ratio
            self.current_cash -= self.last_margin + self.transaction_cost
        elif action == FLAT and self.current_position in (LONG, SHORT):
            # close position
            self.current_cash += self.last_margin + self.current_position_pnl
            self.current_cash += self.transaction_cost  # correct for double substract
            self.last_margin = 0
            self.current_position_pnl = 0
        elif (action == LONG and self.current_position == SHORT) or (action == SHORT and self.current_position == LONG):
            # close position
            self.current_cash += self.last_margin + self.current_position_pnl
            self.current_cash += self.transaction_cost  # correct for double substract
            # then open new position
            pnl -= self.transaction_cost
            self.current_position_pnl = -self.transaction_cost
            self.last_margin = self.unit * self.current_price * self.ratio
            self.current_cash -= self.last_margin + self.transaction_cost


        self.total_pnl += pnl

        # update reward
        reward = self.get_reward(action)

        # update position
        self.current_position = action #STATE_MATHINE[(current_position, action)]

        # increment time step
        self.current_step += 1

        # update price track
        self.last_price = self.current_price
        self.current_price = self.data.iloc[self.current_step, 3]

        observation = self.get_observation()
        done = self.is_trading_done()
        info = {}  # modify later

        return observation, reward, done, info

    def get_reward(self, action):
        # mu = 1 #  a fixed number per contract at each trade
        #reward = mu * sigma_target * (action/sigma*r - bp * price * abs(action/sigma-prev_action/prev_sigma))
        sigma = self.data.close.iloc[self.current_step-60:self.current_step].std()
        price_change = self.current_price - self.last_price
        reward = self.sigma_target * (action/sigma*price_change - self.transaction_cost)
        return reward

    def is_trading_done(self) -> bool:
        done = False
        # check whether account goes broke
        minimal_margin = self.current_price * self.unit * self.ratio
        if self.current_cash <= minimal_margin:
            done = True
        # check whether simulation is done
        if self.current_step >= self.total_period - 1:
            done = True
        return done

    def get_observation(self) -> np.array:
        """Get the observation matrix
        There are two components of the matrix, the asset_matrix and position_matrix,
        Shape of asset_matrix is (time_step,features), and shape of position_matrix is
        (time_step, 1).
        Returns:
            numpy.array: Return the concatenated matrix, which is (time_step, features+1)
        """
        # transformed_obs = self._transformer.transform(self.data.iloc[self.current_step-self.history_length:self.current_step])

        observation = (self.transformed_obs[self.current_step-self.history_length:self.current_step, :], self.current_position)
        return observation

    def _render(self, mode='human'):
        """Render the environment
        """
        pass

class FeatureTransformer:
    def __init__(self, X):
        self.scaler = preprocessing.MinMaxScaler()
        self.scaler.fit(X)

    def transform(self, data):
        return self.scaler.transform(data) # .to_frame().T
