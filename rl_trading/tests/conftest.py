import pytest
from rl_trading.envs.market_env import MarketEnv

@pytest.fixture
def market_env():
    kwargs = {'cash': 1e5,
              'unit': 5,
              'ratio': 0.07,
              'history_length': 10,
              'start_step': 100}
    env = MarketEnv(**kwargs)
    yield env
