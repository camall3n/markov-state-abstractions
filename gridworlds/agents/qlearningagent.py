from collections import defaultdict
import random
from .baseagent import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(self, actions, alpha=0.01, epsilon=0.1, gamma=0.99, skills=None):
        super().__init__(actions, alpha, epsilon, gamma, skills)
        self.default_q = 0.0
        self.q_table = defaultdict(lambda : defaultdict(lambda: self.default_q))
        self.Q = lambda s, a: self.q_table[s][a]

    def act(self, observation, reward, learning=True):
        rep = self.abstract(observation)
        self.skill_reward += reward

        # Check if current skill should terminate
        if self.skills and self.running_skill:
            _, _, term = self.skills[self.running_skill]()
            if term:
                self.prev_action = self.running_skill
                self.running_skill = None

        # Learning update
        if learning and self.prev_rep:
            if not self.running_skill:
                self.update(self.prev_rep, self.prev_action, self.skill_reward, rep)
                self.skill_reward = 0

        # Action selection
        if not self.running_skill:
            # Epsilon-greedy selection w.r.t. valid actions/skills
            valid_choices = self.get_valid_skills() if self.skills else self.actions
            if random.random() < self.epsilon:
                choice = random.choice(valid_choices)
            else:
                choice = self.argmax_q(rep, valid_choices)

            if self.skills:
                self.running_skill = choice
            else:
                base_action = choice
                self.prev_action = base_action
            self.prev_rep = rep

        # Unpack base-level action if necessary
        if self.skills:
            _, base_action, _ = self.skills[self.running_skill]()

        return base_action

    def get_valid_skills(self):
        skill_info = [(name, skill()) for name, skill in self.skills.items()]
        valid_skills = [skill for (skill, (can_run, _, _)) in skill_info if can_run]
        return valid_skills

    def argmax_q(self, rep, actions):
        return max(actions, key=lambda a: self.Q(rep, a))

    def max_q(self, rep, actions):
        return max([self.Q(rep, skill) for skill in actions])

    def update(self, prev_rep, action, reward, rep):
        s = prev_rep
        a = action
        r = reward
        s_next = rep
        if self.skills:
            actions = self.get_valid_skills()
        else:
            actions = self.actions
        max_q_next = self.max_q(s_next, actions)
        q_sa = self.Q(s, a)
        self.q_table[s][a] = (1-self.alpha) * q_sa + self.alpha * (r + self.gamma * max_q_next)
