import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv

class Actions(Enum):
    Sell = 0
    Buy = 1
    Hold = 2
    Close = 3


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, amount, leverage, window_size, frame_bound):
        assert df.ndim == 2
        assert len(frame_bound) == 2
        self.frame_bound = frame_bound
        #super().__init__(df, window_size)
        self.amount = amount
        self.leverage = leverage
        self.trade_price = 0
        self.Position =False
        self.Position2=0
        self.seed()
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        return self._get_observation()

    def step(self, action):
        step_reward = 0
        step_profit = 0
        current_price = self.prices[self._current_tick]
        trade_price = self.trade_price
        self._done = False
        self._current_tick += 1
        if self._current_tick == self._end_tick:
            price_diff = current_price - trade_price
            if self.Position2 == 1:
                if price_diff > 0:
                    #print("reward pos")
                    step_reward = np.abs(price_diff)
                    step_profit = (np.abs(price_diff) / current_price) * self.amount * self.leverage
                if price_diff < 0:
                    a=0
                    #print("reward neg")
            if self.Position2 == 2:
                if price_diff < 0:
                    #print("reward pos")
                    step_reward = np.abs(price_diff)
                    step_profit = (np.abs(price_diff) / current_price) * self.amount * self.leverage
                if price_diff > 0:
                    a = 0
                    #print("reward neg")
            self._done = True
        # step_reward = self._calculate_reward(action)
        # self._total_reward += step_reward
        # self._update_profit(action)

        if action == Actions.Buy.value and self.Position == False:
            #print("BUY")
            self.trade_price = current_price
            self.Position = True
            self.Position2 = 1
        if action == Actions.Sell.value and self.Position == False:
            #print("SELL")
            self.trade_price = current_price
            self.Position = True
            self.Position2 = 2
        if action == Actions.Hold.value and self.Position == False:
            a = 0
            #print("Hold")
        if action == Actions.Close.value and self.Position == True:
            #print("Close")
            price_diff = current_price - trade_price
            if self.Position2 == 1:
                if price_diff > 0:
                    #print("reward pos")
                    step_reward = np.abs(price_diff)
                    step_profit = (np.abs(price_diff) / current_price) * self.amount * self.leverage
                if price_diff < 0:
                    a = 0
                    #print("reward neg")
            if self.Position2 == 2:
                if price_diff < 0:
                    #print("reward pos")
                    step_reward = np.abs(price_diff)
                    step_profit = (np.abs(price_diff) / current_price) * self.amount * self.leverage
                if price_diff > 0:
                    a = 0
                    #print("reward neg")
            self.Position = False
        observation = self._get_observation()
        self._total_reward += step_reward
        self._total_profit += step_profit
        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit
        )
        self._update_history(info)

        return observation, step_reward, self._done, info

    def _get_observation(self):
        return self.signal_features[(self._current_tick - self.window_size):self._current_tick]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self, mode='human'):
        print("Total Reward: %.6f" % self._total_reward + ' ~ ' +
              "Total Profit: %.6f" % self._total_profit)

        plt.pause(0.01)

    def render_all(self, mode='human'):
        print("Total Reward: %.6f" % self._total_reward + ' ~ ' +
              "Total Profit: %.6f" % self._total_profit)

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0] - self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices, signal_features

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _update_profit(self, action):
        raise NotImplementedError

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError
    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs