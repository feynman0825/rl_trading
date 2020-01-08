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


def is_valid_action(position, action):
    '''Check whether action is valid
    '''
    if (action == BUY and position == LONG) or \
        (action == SELL and position == SHORT):
        return False
    return True


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
        self._initial_cash = kwargs['cash']
        self._unit = kwargs['unit']
        self._ratio = kwargs['ratio']
        self._target_profit_factor = kwargs['target_profit_factor']
        self.action_space = np.array([HOLD, BUY, SELL])
        self.observation_space = spaces.Box(0, 1, shape=(7,))
        self._transaction_cost = 15
        self._load_data()
        self._transformer = FeatureTransformer(self._data)
        self.reset()

    def reset(self):
        """Reset the state of the environment and returns an initial observation.
        Returns:
            numpy.array: The initial observation of the space. Initial reward is assumed to be 0.
        """
        self._current_position = FLAT
        self._total_pnl = 0.0
        self._current_cash = self._initial_cash
        self._current_step = 0
        self._total_pnl = 0
        self._current_price = self._data.iloc[0, 3]
        self._last_price = self._current_price
        self._current_position_pnl = 0
        self._last_margin = 0
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
        self._data = cu_df

    @property
    def total_period(self):
        return self._data.shape[0]

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (int): 0 - hold, 1 - buy, 2 - sell
        Returns:
            tuple:
                - observation (numpy.array): Agent's observation of the current environment.
                - reward (float) : Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended, in which case
                               further step() calls will return undefined results.
                - info (str): Contains auxiliary diagnostic information
                              (helpful for debugging, and sometimes learning).
        """
        assert action in self.action_space, "Action should be either 0,1 or 2"
        logger.info('Agent at time step of %d takes %s action' %
                    (self._current_step, action))

        # check action

        # main logic
        current_position = self._current_position
        pnl = 0.0
        reward = 0

        # Make sure action is valid
        if not is_valid_action(current_position, action):
            action = HOLD
            reward += -1

        diff_price = self._current_price - self._last_price
        pnl = self._unit * diff_price * current_position
        self._current_position_pnl += pnl
        if action in (BUY, SELL):
            if self._current_position == FLAT:
                pnl -= self._transaction_cost
                self._current_position_pnl -= self._transaction_cost
                self._last_margin = self._unit * self._current_price * self._ratio
                # update cash
                self._current_cash -= self._last_margin + self._transaction_cost
            else:
                # close position
                self._current_cash += self._last_margin + self._current_position_pnl
                self._current_cash += +self._transaction_cost  # correct for double substract
                self._last_margin = 0
                self._current_position_pnl = 0

        self._total_pnl += pnl

        # update reward
        if pnl == 0:
            reward += 0
        elif pnl > self._target_profit_factor * self._transaction_cost:
            reward += 1
        elif pnl > 0:
            reward += 0.1
        elif pnl < -self._target_profit_factor * self._transaction_cost:
            reward += -1
        else:
            reward += -0.1

        # update position
        self._current_position = STATE_MATHINE[(current_position, action)]

        # increment time step
        self._current_step += 1

        # update price track
        self._last_price = self._current_price
        self._current_price = self._data.iloc[self._current_step, 3]

        observation = self.get_observation()
        done = self._is_trading_done()
        info = {}  # modify later

        return observation, reward, done, info

    def _is_trading_done(self) -> bool:
        done = False
        # check whether account goes broke
        minimal_margin = self._current_price * self._unit * self._ratio
        if self._current_cash <= minimal_margin:
            done = True
        # check whether simulation is done
        if self._current_step >= self.total_period - 1:
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
        # time_step = self._curr_time_step
        # history_length = self._history_length
        # assert time_step >= history_length, 'Time step should be greater than history_length %d' % self._history_length
        # asset_matrix = self._matrix.iloc[(time_step -
        #                                   history_length):time_step].values
        # position_matrix = np.array(
        #     self._position_history)[(time_step -
        #                              history_length):time_step].reshape(-1, 1)
        # observation = np.concatenate((asset_matrix, position_matrix), axis=1)
        transformed_obs = self._transformer.transform(self._data.iloc[self._current_step])
        observation = np.array(transformed_obs.tolist()[0] + [self._current_position])
        return observation

    def _render(self, mode='human'):
        """Render the environment
        """
        pass

class FeatureTransformer:
    def __init__(self, X):
        self.scaler = preprocessing.MinMaxScaler()
        self.scaler.fit(X)

    def transform(self, new_data):
        return self.scaler.transform(new_data.to_frame().T)
