import numpy as np

class RandomAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def reset(self):
        pass

    def act(self, x):
        return np.random.randint(self.n_actions)

    def train(self, x, a, r, xp, done):
        pass
