import gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#%%
class Annotate(gym.Wrapper):
    """
    Description:
        Converts a 3-D pendulum state (y, x, theta_dot) to an annotated state

    Usage:
        Wraps a Pendulum environment to get compressed state information
    """
    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        return self.convert_state(state)

    def step(self, action):
        next_state, reward, done, info = self.env.step([action])
        return self.convert_state(next_state), reward, done, info

    @staticmethod
    def convert_state(state):
        y, x, theta_dot = state
        state = {
            'x': x,
            'y': y,
            'theta': np.arctan2(y, x)/np.pi,
            'theta_dot': theta_dot,
        }
        if state['theta'] > 1/2:
            state['theta'] -= 2
        mass = 1.0
        gravity = 9.81
        radius = 1.0
        velocity = theta_dot * radius
        state['potential_energy'] = 2 * mass * gravity * (y + 1)
        state['kinetic_energy'] = 0.5 * mass * (velocity)**2
        state['energy'] = state['potential_energy'] + state['kinetic_energy']
        return state

#%%
env = gym.make('Pendulum-v0')#, environment_kwargs={'flat_observation': True})
env = Annotate(env)
state = env.reset()

plt.show()
n_episodes = 10
trajectories = []
for i in tqdm(range(n_episodes)):
    done = False
    state = env.reset()
    trajectory = [state]
    imgs = []
    while not done:
        # a = env.action_space.sample()
        # a = 0
        # a = 1.0 # +torque (cw?)
        # a = -1.0 # -torque (ccw?)
        energy_difference = state['energy'] - 9.81*2*2
        if np.abs(energy_difference) > 0.01:
            a = (-np.sign(energy_difference) * np.sign(state['theta_dot']))
        else:
            a = -np.sign(state['x'])
        state, _, done, _ = env.step(a)
        trajectory.append(state)
        # img = env.render(mode='rgb_array', use_opencv_renderer=True)
        # plt.imshow(img)
        # plt.draw()
        # plt.pause(0.001)
    trajectory = {k: np.asarray([dic[k] for dic in trajectory]) for k in trajectory[0]}
    trajectories.append(trajectory)


plt.figure()
for t in trajectories:
    plt.plot(t['kinetic_energy'], t['potential_energy'])
plt.xlabel('KE')
plt.ylabel('PE')
plt.show()

plt.figure()
i = 0
for t in trajectories:
    plt.plot(t['theta_dot'])
plt.ylabel('Velocity')
plt.show()

plt.figure()
i = 0
for t in trajectories:
    energy = t['kinetic_energy'] + t['potential_energy']
    plt.plot(energy)
    # plt.plot(t['theta_dot'])
plt.ylabel('Energy')
plt.show()

#%%

plt.figure()
for t in trajectories:
    energy = t['kinetic_energy'] + t['potential_energy']
    plt.plot(t['theta_dot'], energy)
plt.ylabel('Energy')
plt.xlabel('Velocity')
plt.show()

#%%
plt.figure()
i=2
for t in trajectories:
    theta = np.copy(t['theta'])
    theta[theta<-1.45] += 2
    plt.plot(theta, t['theta_dot'])
plt.ylabel('Velocity')
plt.xlabel('Angle (π radians)')
plt.show()
