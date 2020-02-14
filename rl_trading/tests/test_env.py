
from rl_trading.utils.constant import *
import numpy as np

def test_load_data(market_env):
    assert market_env.data.shape == (233099, 6)

def test_reset(market_env):
    obs = market_env.reset()
    # assert market_env.current_position == FLAT
    assert market_env.second_last_action == FLAT
    assert market_env.total_pnl == 0.0
    assert market_env.ratio == 0.07
    assert market_env.current_cash == market_env.initial_cash
    assert market_env.current_step == 100
    assert market_env.current_price == 28380
    assert market_env.last_price == 28380
    # assert np.allclose(obs[0][0, 0], 0.0897435897)
    assert np.allclose(obs[0, 0], 0.0897435897)

def test_get_observation(market_env):
    obs = market_env.get_observation()
    # assert isinstance(obs, tuple)
    assert isinstance(obs, np.ndarray)
    assert np.allclose(obs[0, 0], 0.0897435897)
    # assert obs[1] == 0


def test_total_period(market_env):
    assert market_env.total_period == 233099

def test_trading_done_not_enough_cash(market_env):
    market_env.current_cash = 100
    assert market_env.is_trading_done()

def test_trading_done_normal(market_env):
    assert not market_env.is_trading_done()

def test_trading_done_exceed_trading_period(market_env):
    market_env.current_step = 1e8
    assert market_env.is_trading_done()


def test_env_step(market_env):
    # do nothing at the first step
    observation, reward, done, info = market_env.step(FLAT)
    # assert isinstance(observation, tuple)
    assert isinstance(observation, np.ndarray)
    assert np.allclose(observation[0, 0], 0.0897435897)
    assert reward == 0
    assert not done
    assert isinstance(info, dict)
    assert market_env.current_step == 101
    assert market_env.current_cash == 1e5
    assert market_env.second_last_action == FLAT
    assert market_env.last_margin == 0
    assert market_env.total_pnl == 0.0
    assert market_env.current_cash == 1e5
    assert market_env.total_pnl == 0
    assert market_env.current_price == 28370
    assert market_env.last_price == 28380


    # buy at step 2
    observation, reward, done, info = market_env.step(LONG)
    # assert isinstance(observation, tuple)
    assert isinstance(observation, np.ndarray)
    # assert np.allclose(reward, -0.3164697097884613)
    assert not done
    assert isinstance(info, dict)
    assert market_env.current_step == 102
    assert market_env.second_last_action == LONG
    assert market_env.last_margin == 28380*5*0.07
    assert market_env.current_cash == 1e5 - 28380*5*0.07 - 15 - 10*5
    assert market_env.total_pnl == -15 - 10*5
    assert market_env.current_price == 28390
    assert market_env.last_price == 28370

    # HOLD at step 3
    observation, reward, done, info = market_env.step(LONG)
    # assert isinstance(observation, tuple)
    assert isinstance(observation, np.ndarray)
    # assert np.allclose(reward, 0.5068436714759406)
    assert not done
    assert isinstance(info, dict)
    assert market_env.current_step == 103
    assert market_env.second_last_action == LONG
    assert market_env.last_margin == 28380*5*0.07
    assert market_env.current_cash == 1e5 - 28380*5*0.07 - 15 - 10*5 + 20*5
    assert market_env.total_pnl == -15 - 10*5 + 20*5
    assert market_env.current_price == 28400
    assert market_env.last_price == 28390
    

    # HOLD at step 4
    observation, reward, done, info = market_env.step(FLAT)
    # assert isinstance(observation, tuple)
    assert isinstance(observation, np.ndarray)
    assert np.allclose(reward, 0)
    assert not done
    assert isinstance(info, dict)
    assert market_env.current_step == 104
    assert market_env.second_last_action == FLAT
    assert market_env.last_margin == 0
    assert market_env.current_cash == 1e5 - 15 - 10*5 + 20*5
    assert market_env.total_pnl == -15 - 10*5 + 20*5
    assert market_env.current_price == 28430
    assert market_env.last_price == 28400

    # HOLD at step 5
    observation, reward, done, info = market_env.step(FLAT)
    # assert isinstance(observation, tuple)
    assert isinstance(observation, np.ndarray)
    assert np.allclose(reward, 0)
    assert not done
    assert isinstance(info, dict)
    assert market_env.current_step == 105
    assert market_env.second_last_action == FLAT
    assert market_env.last_margin == 0
    assert market_env.current_cash == 1e5 - 15 - 10*5 + 20*5
    assert market_env.total_pnl == -15 - 10*5 + 20*5
    assert market_env.current_price == 28430
    assert market_env.last_price == 28430

    # HOLD at step 6
    observation, reward, done, info = market_env.step(SHORT)
    # assert isinstance(observation, tuple)
    assert isinstance(observation, np.ndarray)
    # assert np.allclose(reward, -0.08465526657524954)
    assert not done
    assert isinstance(info, dict)
    assert market_env.current_step == 106
    assert market_env.second_last_action == SHORT
    assert market_env.last_margin == 28430*5*0.07
    assert market_env.current_cash == 1e5 - 15 - 10*5 + 20*5 - 15 - 28430*5*0.07
    assert market_env.total_pnl == -15 - 10*5 + 20*5 - 15
    assert market_env.current_price == 28400
    assert market_env.last_price == 28430

    # HOLD at step 7
    observation, reward, done, info = market_env.step(SHORT)
    # assert isinstance(observation, tuple)
    assert isinstance(observation, np.ndarray)
    # assert np.allclose(reward, 0.8803176887665144)
    assert not done
    assert isinstance(info, dict)
    assert market_env.current_step == 107
    assert market_env.second_last_action == SHORT
    assert market_env.last_margin == 28430*5*0.07
    assert market_env.current_cash == 1e5 - 15 - 10*5 + 20*5 - 15 - 28430*5*0.07 + 30*5
    assert market_env.total_pnl == -15 - 10*5 + 20*5 - 15 + 30*5
    assert market_env.current_price == 28410
    assert market_env.last_price == 28400

    # HOLD at step 8
    observation, reward, done, info = market_env.step(FLAT)
    # assert isinstance(observation, tuple)
    assert isinstance(observation, np.ndarray)
    assert np.allclose(reward, 0)
    assert not done
    assert isinstance(info, dict)
    assert market_env.current_step == 108
    assert market_env.second_last_action == FLAT
    assert market_env.last_margin == 0
    assert market_env.current_cash == 1e5 - 15 - 10*5 + 20*5 - 15 + 30*5
    assert market_env.total_pnl == -15 - 10*5 + 20*5 - 15 + 30*5
    assert market_env.current_price == 28420
    assert market_env.last_price == 28410
