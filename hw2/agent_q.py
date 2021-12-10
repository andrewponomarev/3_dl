import random
import pickle


class Player:
    def __init__(self, name, board_cols, board_rows, exp_rate, lr, decay_gamma):
        self.name = name
        self.states = []
        self.lr = lr
        self.exp_rate = exp_rate
        self.decay_gamma = decay_gamma
        self.states_value = {}
        self.board_cols = board_cols
        self.board_rows = board_rows

    def getHash(self, board):
        return board.tobytes()

    def get_exp_rate(self):
        return self.exp_rate

    def set_exp_rate(self, exp_rate):
        self.exp_rate = exp_rate

    def chooseAction(self, positions, current_board, symbol):
        if random.random() <= self.exp_rate:
            idx = random.choice(range(len(positions)))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardHash = self.getHash(next_board)
                value = random.random() if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                if value >= value_max:
                    value_max = value
                    action = p
        return action

    def addState(self, state):
        self.states.append(state)

    def feedReward(self, reward):
        for st in self.states[::-1]:
            if self.states_value.get(st) is None:
                self.states_value[st] = random.random()
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()