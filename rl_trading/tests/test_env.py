
from rl_trading.utils.constant import *
from rl_trading.envs.market_env import is_valid_action
import numpy as np

def test_load_data(market_env):
    assert market_env._data.shape == (233099, 6)

def test_reset(market_env):
    obs = market_env.reset()
    assert market_env._current_position == FLAT
    assert market_env._total_pnl == 0.0
    assert market_env._ratio == 0.07
    assert market_env._current_cash == market_env._initial_cash
    assert market_env._current_step == 0
    assert market_env._current_price == 29070
    assert market_env._last_price == 29070
    # assert obs[0] == 29170
    assert np.allclose(obs[0], 0.10842038619816396)

def test_get_observation(market_env):
    obs = market_env.get_observation()
    assert isinstance(obs, np.ndarray)
    # assert obs[0] == 29170
    assert np.allclose(obs[0], 0.10842038619816396)
    assert obs[-1] == 0

def test_is_valid_action():
    assert is_valid_action(FLAT, BUY)
    assert is_valid_action(FLAT, SELL)
    assert is_valid_action(FLAT, HOLD)
    assert not is_valid_action(LONG, BUY)
    assert is_valid_action(LONG, SELL)
    assert is_valid_action(LONG, HOLD)
    assert is_valid_action(SHORT, BUY)
    assert not is_valid_action(SHORT, SELL)
    assert is_valid_action(SHORT, HOLD)

def test_total_period(market_env):
    assert market_env.total_period == 233099

def test_trading_done_not_enough_cash(market_env):
    market_env._current_cash = 100
    assert market_env._is_trading_done()

def test_trading_done_normal(market_env):
    assert not market_env._is_trading_done()

def test_trading_done_exceed_trading_period(market_env):
    market_env._current_step = 1e8
    assert market_env._is_trading_done()


def test_env_step(market_env):
    # do nothing at the first step
    observation, reward, done, info = market_env.step(FLAT)
    assert isinstance(observation, np.ndarray)
    # assert observation[0] == 29090
    assert np.allclose(observation[0], 0.10715416270971828)
    assert reward == 0
    assert not done
    assert isinstance(info, dict)
    assert market_env._current_step == 1
    assert market_env._current_cash == 1e5
    assert market_env._current_position == FLAT
    assert market_env._last_margin == 0
    assert market_env._total_pnl == 0.0
    assert market_env._current_cash == 1e5
    assert market_env._total_pnl == 0
    assert market_env._current_price == 28970
    assert market_env._last_price == 29070
    assert market_env._current_position_pnl == 0


    # buy at step 2
    observation, reward, done, info = market_env.step(BUY)
    assert isinstance(observation, np.ndarray)
    # assert observation[0] == 29060
    assert reward == -0.1
    assert not done
    assert isinstance(info, dict)
    assert market_env._current_step == 2
    assert market_env._current_position == LONG
    assert market_env._last_margin == 28970*5*0.07
    assert market_env._current_cash == 1e5 - 15 - 28970*5*0.07
    assert market_env._total_pnl == -15
    assert market_env._current_price == 29030
    assert market_env._last_price == 28970
    assert market_env._current_position_pnl == -15

    # HOLD at step 3
    observation, reward, done, info = market_env.step(HOLD)
    assert isinstance(observation, np.ndarray)
    # assert observation[0] == 29100
    assert reward == 1
    assert not done
    assert isinstance(info, dict)
    assert market_env._current_step == 3
    assert market_env._current_position == LONG
    assert market_env._last_margin == 28970*5*0.07
    assert market_env._current_cash == 1e5 - 15 - 28970*5*0.07
    assert market_env._total_pnl == 285
    assert market_env._current_price == 29060
    assert market_env._last_price == 29030
    assert market_env._current_position_pnl == 285
    

    # HOLD at step 4
    observation, reward, done, info = market_env.step(SELL)
    assert isinstance(observation, np.ndarray)
    # assert observation[0] == 29150
    assert reward == 0.1
    assert not done
    assert isinstance(info, dict)
    assert market_env._current_step == 4
    assert market_env._current_position == FLAT
    assert market_env._last_margin == 0
    assert market_env._current_cash == 1e5 + 285 + 30*5
    assert market_env._total_pnl == 285 + 30*5
    assert market_env._current_price == 29130
    assert market_env._last_price == 29060
    assert market_env._current_position_pnl == 0

    # HOLD at step 5
    observation, reward, done, info = market_env.step(HOLD)
    assert isinstance(observation, np.ndarray)
    # assert observation[0] == 29120
    assert reward == 0
    assert not done
    assert isinstance(info, dict)
    assert market_env._current_step == 5
    assert market_env._current_position == FLAT
    assert market_env._last_margin == 0
    assert market_env._current_cash == 1e5 + 285 + 30*5
    assert market_env._total_pnl == 285 + 30*5
    assert market_env._current_price == 29090
    assert market_env._last_price == 29130
    assert market_env._current_position_pnl == 0

    # HOLD at step 6
    observation, reward, done, info = market_env.step(SELL)
    assert isinstance(observation, np.ndarray)
    # assert observation[0] == 29110
    assert reward == -0.1
    assert not done
    assert isinstance(info, dict)
    assert market_env._current_step == 6
    assert market_env._current_position == SHORT
    assert market_env._last_margin == 29090*5*0.07
    assert market_env._current_cash == 1e5 + 285 + 30*5 - 29090*5*0.07 - 15
    assert market_env._total_pnl == 285 + 30*5 - 15
    assert market_env._current_price == 29050
    assert market_env._last_price == 29090
    assert market_env._current_position_pnl == - 15

    # HOLD at step 7
    observation, reward, done, info = market_env.step(SELL)
    assert isinstance(observation, np.ndarray)
    # assert observation[0] == 29070
    assert reward == -1 + 1
    assert not done
    assert isinstance(info, dict)
    assert market_env._current_step == 7
    assert market_env._current_position == SHORT
    assert market_env._last_margin == 29090*5*0.07
    assert market_env._current_cash == 1e5 + 285 + 30*5 - 29090*5*0.07 - 15
    assert market_env._total_pnl == 285 + 30*5 - 15 + 40*5
    assert market_env._current_price == 29060
    assert market_env._last_price == 29050
    assert market_env._current_position_pnl == - 15 + 40*5

    # HOLD at step 8
    observation, reward, done, info = market_env.step(BUY)
    assert isinstance(observation, np.ndarray)
    # assert observation[0] == 29100
    assert reward == -0.1
    assert not done
    assert isinstance(info, dict)
    assert market_env._current_step == 8
    assert market_env._current_position == FLAT
    assert market_env._last_margin == 0
    assert market_env._current_cash == 1e5 + 285 + 30*5 - 15 + 40*5 - 10*5
    assert market_env._total_pnl == 285 + 30*5 - 15 + 40*5 - 10*5
    assert market_env._current_price == 29100
    assert market_env._last_price == 29060
    assert market_env._current_position_pnl == 0