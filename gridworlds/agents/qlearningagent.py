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

        if self.skills:
            base_action = None
            while base_action is None:
                if not self.current_skill:
                    if learning and self.prev_rep:
                        self.update(self.prev_rep, self.prev_action, self.skill_reward, rep)
                        self.skill_reward = 0

                    # Choose a new skill using epsilon-greedy w.r.t. valid skills
                    valid_skills = self.get_valid_skills()
                    if random.random() < self.epsilon:
                        skill_choice = random.choice(valid_skills)
                    else:
                        skill_choice = self.argmax_q(rep, valid_skills)
                    self.current_skill = skill_choice
                    self.prev_rep = rep

                # Compute next base-level action for current skill
                _, base_action, term = self.skills[self.current_skill]()
                if term:
                    self.prev_action = self.current_skill
                    self.current_skill = None
        else:
            if learning and self.prev_rep:
                self.update(self.prev_rep, self.prev_action, reward, rep)

            if random.random() < self.epsilon:
                action = random.choice(self.actions)
            else:
                action = self.argmax_q(rep, self.actions)
            base_action = action
            self.prev_action = base_action
            self.prev_rep = rep

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
