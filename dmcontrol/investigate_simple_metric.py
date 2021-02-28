from tqdm import tqdm
# ---------------------------------------------------------------
# This hack prevents a matplotlib/dm_control crash on macOS.
# We open/close a plot before importing dm_control.suite, which
# in this case happens when we import gym and make a dm2gym env.
import matplotlib.pyplot as plt
plt.plot()
plt.close()
import gym
# ---------------------------------------------------------------

import imageio
import numpy as np

from dmcontrol import gym_wrappers as wrap
from dmcontrol import rad

#%%

class DMCPendulum2D(gym.Wrapper):
    """
    Description:
        Converts the default 3-D pendulum state (x, y, theta_dot) to a 2-D state (theta, theta_dot)

    Usage:
        Wraps a dm2gym Pendulum environment to get compressed state information
    """
    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        return self.convert_state(state)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return self.convert_state(next_state), reward, done, info

    @staticmethod
    def convert_state(state):
        theta_dot = state['velocity'][0]
        y, x = state['orientation']
        theta = np.arctan2(y, x)
        return np.array([theta, theta_dot])


class DMCPendulumEnergy(gym.Wrapper):
    """
    Description:
        Converts a 3-D pendulum state (x, y, theta_dot) to a 2-D energy state (potential, kinetic)

    Usage:
        Wraps a dm2gym Pendulum environment to get compressed state information
    """
    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        return self.convert_state(state)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return self.convert_state(next_state), reward, done, info

    @staticmethod
    def convert_state(state):
        theta_dot = state['velocity'][0]
        y, x = state['orientation']
        mass = 1.0
        gravity = 9.8
        radius = 1.0
        velocity = theta_dot * radius
        potential_energy = 2 * mass * gravity * (y + 1)
        kinetic_energy = 0.5 * mass * (velocity)**2
        return np.array([potential_energy, kinetic_energy])

class DMCPendulumAnnotated(gym.Wrapper):
    """
    Description:
        Converts a 3-D pendulum state (x, y, theta_dot) to a 2-D energy state (potential, kinetic)

    Usage:
        Wraps a dm2gym Pendulum environment to get compressed state information
    """
    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        return self.convert_state(state)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return self.convert_state(next_state), reward, done, info

    @staticmethod
    def convert_state(state):
        theta_dot = state['velocity'][0]
        y, x = state['orientation']
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

def wrap_env(env, feature_type='visual', size=(84, 84), action_repeat=1, frame_stack=3):
    env = wrap.FixedDurationHack(env)
    env = wrap.ObservationDictToInfo(env, "observations")

    if feature_type == 'expert':
        env = wrap.MaxAndSkipEnv(env, skip=action_repeat, max_pool=False)
    else:
        env = wrap.RenderOpenCV(env)
        env = wrap.Grayscale(env)
        env = wrap.ResizeObservation(env, size)
        env = wrap.MaxAndSkipEnv(env, skip=action_repeat, max_pool=False)
        env = wrap.FrameStack(env, frame_stack, lazy=False)
    return env


env = gym.make('dm2gym:PendulumSwingup-v0')#, environment_kwargs={'flat_observation': True})
env = env.env
# env = DMCPendulum2D(env)
env = DMCPendulumEnergy(env)

state = env.reset()
state = env.step(env.action_space.sample())[0]
img = env.render(mode='rgb_array', use_opencv_renderer=True)
plt.imshow(img)

plt.ion()
plt.show()

#%%
env = gym.make('Pendulum-v0')#, environment_kwargs={'flat_observation': True})
# env = env.env
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


# t = trajectories[0]
# pe, ke = t['potential_energy'], t['kinetic_energy']
# plt.plot(pe+ke, '.-k', label='PE+KE')
# plt.plot(pe, label='PE')
# plt.plot(ke, '--', label='KE')
# plt.xlabel('Time')
# plt.ylabel('Energy')
# plt.legend()
# plt.show()

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
