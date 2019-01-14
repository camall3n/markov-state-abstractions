
class BaseAgent:
    def __init__(self, actions, alpha=0.01, epsilon=0.1, gamma=0.99, skills=None):
        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.prev_rep = None
        self.prev_action = None
        self.running_action = None
        self.action_reward = 0

    def end_of_episode(self):
        self.prev_rep = None
        self.prev_action = None
        self.running_action = None
        self.action_reward = 0

    def abstract(self, observation):
        rep = tuple(observation)
        return rep

    def act(self, observation, reward, learning=True):
        rep = self.abstract(observation)
        self.action_reward += reward

        if self.running_action:
            self.clear_terminated_actions()

        # Learning update
        if learning and (not self.running_action) and (self.prev_action is not None):
            self.update(self.prev_rep, self.prev_action, self.action_reward, rep)
            self.action_reward = 0

        # Action selection
        if not self.running_action:
            self.select_next_action(rep)

        base_action = self.unpack_running_action()
        return base_action

    def clear_terminated_actions(self):
        self.running_action = None

    def get_valid_actions(self):
        return self.actions

    def unpack_running_action(self):
        return self.running_action
