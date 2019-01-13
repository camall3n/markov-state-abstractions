
class BaseAgent:
    def __init__(self, actions, alpha=0.01, epsilon=0.1, gamma=0.99, skills=None):
        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.skills = skills

        self.prev_rep = None
        self.prev_action = None
        self.current_skill = None
        self.skill_reward = 0

    def end_of_episode(self):
        self.prev_rep = None
        self.prev_action = None
        self.current_skill = None
        self.skill_reward = 0
