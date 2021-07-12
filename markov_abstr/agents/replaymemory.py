from collections import namedtuple
import random

import numpy as np

# Adapted from Pytorch docs

Experience = namedtuple('Experience', ('x', 'a', 'r', 'xp', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, itemize=False):
        result = random.sample(self.memory, batch_size)
        if itemize:
            result = zip(*result)
            result = map(np.asarray, result)
            result = Experience(*result)
            result = tuple(result)
            x, a, r, xp, d = result
            r = np.expand_dims(r, axis=-1)
            d = np.expand_dims(d, axis=-1)
            result = x, a, r, xp, d
        return result

    def get_last(self, batch_size):
        return self.memory[-batch_size:]

    def __len__(self):
        return len(self.memory)
