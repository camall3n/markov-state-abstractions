from collections import namedtuple
import random

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

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_last(self, batch_size):
        return self.memory[-batch_size:]

    def __len__(self):
        return len(self.memory)
