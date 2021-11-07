from .strategy import BaseStrategy
from .environment import *


class BlackJackSimulation:

    def __init__(self, env: BaseBlackJackEnv, num_of_episodes: int, strategy: BaseStrategy):
        self.env = env
        self.num_of_episodes = num_of_episodes
        self.strategy = strategy

    def episode(self):
        state: State = self.env.reset()
        episode_reward = 0
        history = []
        finished = False

        while not finished:
            self.strategy.state_count[state] += 1
            action = self.strategy.action(state)
            new_state, reward, finished, desc = self.env.step(action)
            history.append((state, action, reward))
            state = new_state
            episode_reward += reward

        self.strategy.update_q(history)
        return episode_reward

    def experiment(self):
        mean_rewards = []
        win_rates = []
        wins = 0
        total_reward = 0

        for i in range(1, self.num_of_episodes + 1):
            episode_reward = self.episode()
            if episode_reward > 0:
                wins += 1
            total_reward += episode_reward
            mean_rewards.append(total_reward / i)
            win_rates.append(wins / i)

        return mean_rewards, win_rates


