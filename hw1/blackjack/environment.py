import gym

from abc import abstractmethod
from gym import spaces
from gym.utils import seeding
from dataclasses import dataclass

from .utils import *


@dataclass(frozen=True)
class State:
    player_total: int
    dealers_showing_card: int
    usable_ace: bool


@dataclass(frozen=True)
class CountState(State):
    sum: int


class BaseBlackJackEnv(gym.Env):

    def __init__(self, natural=False, sab=False):
        self.seed()
        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Flag for full agreement with the (Sutton and Barto, 2018) definition. Overrides self.natural
        self.sab = sab
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @abstractmethod
    def step(self, action):
        pass

    def _hit(self):
        # hit: add a card to players hand and return
        self.player.append(draw_card(self.np_random))
        if is_bust(self.player):
            done = True
            reward = -1.0
        else:
            done = False
            reward = 0.0
        return done, reward

    def _stand(self):
        # stick: play out the dealers hand, and score
        done = True
        while sum_hand(self.dealer) < 17:
            self.dealer.append(draw_card(self.np_random))
        reward = cmp(score(self.player), score(self.dealer))
        if self.sab and is_natural(self.player) and not is_natural(self.dealer):
            # Player automatically wins. Rules consistent with S&B
            reward = 1.0
        elif (
                not self.sab
                and self.natural
                and is_natural(self.player)
                and reward == 1.0
        ):
            # Natural gives extra points, but doesn't autowin. Legacy implementation
            reward = 1.5
        return done, reward

    def _get_obs(self):
        player_total = sum_hand(self.player)
        dealers_showing_card = self.dealer[0]
        return State(player_total, dealers_showing_card, usable_ace(self.player))

    def reset(self):
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        return self._get_obs()


class SimpleBlackJackEnv(BaseBlackJackEnv):
    """Simple blackjack environment
   Blackjack is a card game where the goal is to obtain cards that sum to as
   near as possible to 21 without going over.  They're playing against a fixed
   dealer.
   Face cards (Jack, Queen, King) have point value 10.
   Aces can either count as 11 or 1, and it's called 'usable' at 11.
   This game is placed with an infinite deck (or with replacement).
   The game starts with dealer having one face up and one face down card, while
   player having two face up cards. (Virtually for all Blackjack games today).
   The player can request additional cards (hit=1) until they decide to stop
   (stick=0) or exceed 21 (bust).
   After the player sticks, the dealer reveals their facedown card, and draws
   until their sum is 17 or greater.  If the dealer goes bust the player wins.
   If neither player nor dealer busts, the outcome (win, lose, draw) is
   decided by whose sum is closer to 21.  The reward for winning is +1,
   drawing is 0, and losing is -1.
   The observation of a 3-tuple of: the players current sum,
   the dealer's one showing card (1-10 where 1 is ace),
   and whether or not the player holds a usable ace (0 or 1).
   This environment corresponds to the version of the blackjack problem
   described in Example 5.1 in Reinforcement Learning: An Introduction
   by Sutton and Barto.
   http://incompleteideas.net/book/the-book-2nd.html
   """

    def __init__(self, natural=False, sab=False):
        super().__init__(natural, sab)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2))
        )

    def step(self, action):
        assert self.action_space.contains(action)
        if action:
            done, reward = self._hit()
        else:
            done, reward = self._stand()
        return self._get_obs(), reward, done, {}


class DoubleBlackJackEnv(BaseBlackJackEnv):

    def __init__(self, natural=False, sab=False):
        super().__init__(natural, sab)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32),
             spaces.Discrete(11),
             spaces.Discrete(72),
             spaces.Discrete(2))
        )
        self.action_space = spaces.Discrete(3)

    def _double(self):
        self.player.append(draw_card(self.np_random))
        if sum_hand(self.dealer) < 17:
            self.dealer.append(draw_card(self.np_random))
        reward = cmp(score(self.player), score(self.dealer)) * 2
        return True, reward

    def step(self, action):
        assert self.action_space.contains(action)
        if action:
            done, reward = self._hit()
        elif action == 0:
            done, reward = self._stand()
        else:
            done, reward = self._double()
        return self._get_obs(), reward, done, {}


class DoubleCountBlackJackEnv(DoubleBlackJackEnv):
    DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
    COUNT_DICT = {
        1: -1,
        2: 0.5,
        3: 1,
        4: 1,
        5: 1.5,
        6: 1,
        7: 0.5,
        8: 0,
        9: -0.5,
        10: -1
    }

    def __init__(self,
                 n_decks=1,
                 natural=False,
                 sab=False):
        super().__init__(natural, sab)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32),
             spaces.Discrete(11),
             spaces.Discrete(72),
             spaces.Discrete(2))
        )
        self.sum = 0
        self.card_value_dict = DoubleCountBlackJackEnv.COUNT_DICT
        self.deck = DoubleCountBlackJackEnv.DECK.copy() * n_decks

    def draw_card(self):
        card = int(self.np_random.choice(self.deck))
        self.count_card(card)
        self.deck.remove(card)
        return card

    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]

    def count_card(self, card):
        self.sum += self.card_value_dict[card]

    def _get_obs(self):
        return CountState(sum_hand(self.player), self.dealer[0], usable_ace(self.player), sum=self.sum)

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 1:
            done, reward = self._hit()
        elif action == 0:
            done, reward = self._stand()
        else:
            done, reward = self._double()

        if len(self.deck) <= 15:
            played_cards = self.player.copy()
            played_cards.extend(self.dealer)
            self.sum = 0
            self.deck = [x for x in DoubleCountBlackJackEnv.DECK if x not in played_cards]

        return self._get_obs(), reward, done, {}

    def reset(self):
        self.dealer = self.draw_hand()
        self.player = self.draw_hand()
        return self._get_obs()
