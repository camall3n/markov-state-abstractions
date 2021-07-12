from collections import defaultdict
import random
from .baseagent import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(self, actions, alpha=0.01, epsilon=0.1, gamma=0.99):
        super().__init__(actions, alpha, epsilon, gamma)
        self.default_q = 0.0
        self.q_table = defaultdict(lambda : defaultdict(lambda: self.default_q))
        self.Q = lambda s, a: self.q_table[s][a]

    def argmax_q(self, rep, actions):
        return max(actions, key=lambda a: self.Q(rep, a))

    def max_q(self, rep, actions):
        return max([self.Q(rep, skill) for skill in actions])

    def select_next_action(self, rep):
        # Epsilon-greedy selection w.r.t. valid actions/skills
        actions = self.get_valid_actions()
        if random.random() < self.epsilon:
            action = random.choice(actions)
        else:
            action = self.argmax_q(rep, actions)
        self.running_action = action
        self.prev_action = action
        self.prev_rep = rep

    def update(self, prev_rep, action, reward, rep):
        s = prev_rep
        a = action
        r = reward
        s_next = rep
        actions = self.get_valid_actions()
        max_q_next = self.max_q(s_next, actions)
        q_sa = self.Q(s, a)
        self.q_table[s][a] = (1-self.alpha) * q_sa + self.alpha * (r + self.gamma * max_q_next)

class SkilledQLearningAgent(QLearningAgent):
    def __init__(self, options, alpha=0.01, epsilon=0.1, gamma=0.99):
        super().__init__(actions=options, alpha=alpha, epsilon=epsilon, gamma=gamma)

    def clear_terminated_actions(self):
        _, _, term = self.actions[self.running_action]()
        if term: self.running_action = None

    def get_valid_actions(self):
        skill_info = [(name, skill()) for name, skill in self.actions.items()]
        valid_skills = [skill for (skill, (can_run, _, _)) in skill_info if can_run]
        return valid_skills

    def unpack_running_action(self):
        _, base_action, _ = self.actions[self.running_action]()
        return base_action
