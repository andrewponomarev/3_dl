from collections import defaultdict
from abc import abstractmethod
import numpy as np

from .environment import State


class BaseStrategy:

    def __init__(self):
        self.state_count = defaultdict(float)

    @abstractmethod
    def action(self, state: State):
        pass

    @abstractmethod
    def update_q(self, history):
        pass


class RandomStrategy(BaseStrategy):

    def __init__(self, action_num):
        super().__init__()
        self.action_num = action_num

    def action(self, state: State):
        return np.random.randint(0, self.action_num)

    def update_q(self, history):
        pass


class ConstantStrategy(BaseStrategy):

    def __init__(self):
        super().__init__()

    def action(self, state: State):
        if state.player_total <= 18:
            return 1
        else:
            return 0

    def update_q(self, history):
        pass


class MonteCarloControl(BaseStrategy):
    def __init__(self, epsilon, gamma, action_num):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.action_num = action_num
        self.q = defaultdict(lambda: np.zeros(self.action_num))
        self.state_action_count = defaultdict(float)

    def action(self, state):
        action_random = np.random.randint(0, self.action_num)
        action_mc = np.argmax(self.q[state])
        return np.random.choice([action_random, action_mc],
                                p=[1 - self.epsilon, self.epsilon])

    def update_q(self, history):
        g = 0
        for state, action, reward in reversed(history):
            self.state_action_count[(state, action)] += 1
            g = reward + g * self.gamma
            self.q[state][action] += \
                (g - self.q[state][action]) / \
                (self.state_action_count[(state, action)])
