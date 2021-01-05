from collections import deque

import numpy as np
# import torchvision.transforms as T
import gym
import cv2

class IndexedObservation(gym.ObservationWrapper):
    """
    Description:
        Return elements of observation at given indices

    Usage:
        For example, say the base env has observations Box(4) and you want the indices 1 and 3. You
        would pass in indices=[1,3] and the observation_space of the wrapped env would be Box(2).

    Notes:
        - This currently only supports 1D observations but can easily be extended to support
          multidimensional observations
    """
    def __init__(self, env, indices):
        super(IndexedObservation, self).__init__(env)
        self.indices = indices

        assert len(env.observation_space.shape) == 1, env.observation_space
        wrapped_obs_len = env.observation_space.shape[0]
        assert len(indices) <= wrapped_obs_len, indices
        assert all(i < wrapped_obs_len for i in indices), indices
        self.observation_space = gym.spaces.Box(low=env.observation_space.low[indices],
                                                high=env.observation_space.high[indices],
                                                dtype=env.observation_space.dtype)

    def observation(self, observation):
        return observation[self.indices]

# Adapted from https://github.com/openai/gym/blob/master/gym/wrappers/resize_observation.py
class ResizeObservation(gym.ObservationWrapper):
    """
    Description:
        Downsample the image observation to a given shape.

    Usage:
        Pass in requisite shape (e.g. 84,84) and it will use opencv to resize the observation to
        that shape

    Notes:
        - N/A
    """
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape
        self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.resize(observation, self.shape[::-1], interpolation=cv2.INTER_AREA)
        return observation

class ObservationDictToInfo(gym.Wrapper):
    """
    Description:
        Given an env with an observation dict, extract the given state key as the state and pass the
        existing dict into the info.

    Usage:
        Wrap any Dict observation.

    Notes:
        - By convention, no info is return on reset, so that dict is lost.
    """
    def __init__(self, env, state_key):
        gym.Wrapper.__init__(self, env)
        assert type(env.observation_space) == gym.spaces.Dict
        self.observation_space = env.observation_space.spaces[state_key]
        self.state_key = state_key

    def reset(self, **kwargs):
        next_state_as_dict = self.env.reset(**kwargs)
        return next_state_as_dict[self.state_key]

    def step(self, action):
        next_state_as_dict, reward, done, info = self.env.step(action)
        info.update(next_state_as_dict)
        return next_state_as_dict[self.state_key], reward, done, info

class RenderOpenCV(gym.Wrapper):
    """
    Description:
        Given an env that can render with the flag 'use_opencv_renderer',
        call the renderer, and return the rgb_array observation. Also adds
        the original state as an entry to the info dict.

    Usage:
        Wraps a dm2gym environment to get visual features

    Notes:
        - By convention, no info is return on reset, so that state is lost.
    """
    def reset(self, **kwargs):
        _ = self.env.reset(**kwargs)
        next_ob = self.env.render(mode='rgb_array', use_opencv_renderer=True)
        return next_ob

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        info.update({'state': next_state})
        next_ob = self.env.render(mode='rgb_array', use_opencv_renderer=True)
        return next_ob, reward, done, info

class FixedDurationHack(gym.Wrapper):
    """
    Description:
        Given a TimeLimit-wrapped environment, copy the _max_episode_steps
        variable to the unwrapped environment for easier access, in case
        *this* environment gets wrapped by additional wrappers.
    """
    def __init__(self, env):
        super().__init__(env)
        self.unwrapped._max_episode_steps = env._max_episode_steps

class Grayscale(gym.Wrapper):
    """
    Description:
        Given an env with RGB image observations, return grayscale frames.

    Usage:
        Wraps an RGB image-based environment
    """
    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        return cv2.cvtColor(ob, cv2.COLOR_RGB2GRAY)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        return cv2.cvtColor(ob, cv2.COLOR_RGB2GRAY), reward, done, info

class ResetARI(gym.Wrapper):
    """
    Description:
        On reset and step, grab the values of the labeled dict from info and return as state.

    Usage:
        Wrap over ARI env.

    Notes:
        - N/A
    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        # change the observation space to accurately represent
        # the shape of the labeled RAM observations
        self.observation_space = gym.spaces.Box(
            0,
            255,  # max value
            shape=(len(self.env.labels()), ),
            dtype=np.uint8)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        # reset the env and get the current labeled RAM
        return np.array(list(self.env.labels().values()))

    def step(self, action):
        # we don't need the obs here, just the labels in info
        _, reward, done, info = self.env.step(action)
        # grab the labeled RAM out of info and put as next_state
        next_state = np.array(list(info['labels'].values()))
        return next_state, reward, done, info

# Adapted from OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
# class AtariPreprocess(gym.Wrapper):
#     """
#     Description:
#         Preprocessing as described in the Nature DQN paper (Mnih 2015)

#     Usage:
#         Wrap env around this. It will use torchvision to transform the image according to Mnih 2015

#     Notes:
#         - Should be decomposed into using separate envs for each.
#     """
#     def __init__(self, env, shape=(84, 84)):
#         gym.Wrapper.__init__(self, env)
#         self.shape = shape
#         self.transforms = T.Compose([
#             T.ToPILImage(mode='YCbCr'),
#             T.Lambda(lambda img: img.split()[0]),
#             T.Resize(self.shape),
#             T.Lambda(lambda img: np.array(img, copy=False)),
#         ])
#         self.observation_space = gym.spaces.Box(
#             low=0,
#             high=255,
#             shape=self.shape,
#             dtype=np.uint8,
#         )

#     def reset(self, **kwargs):
#         return self.transforms(self.env.reset(**kwargs))

#     def step(self, action):
#         next_state, reward, done, info = self.env.step(action)
#         return self.transforms(next_state), reward, done, info

class MaxAndSkipEnv(gym.Wrapper):
    """
    Description:
        Return only every `skip`-th frame. Repeat action, sum reward, and
        (by default) max over last observations.

    Usage:
        Wrap env and provide skip param.

    Notes:
        - N/A
    """
    def __init__(self, env, skip=4, max_pool=True):
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2, ) + env.observation_space.shape,
                                    dtype=env.observation_space.dtype)
        self._skip = skip
        self._max_pool = max_pool
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        if self._max_pool:
            frame = self._obs_buffer.max(axis=0)
        else:
            frame = self._obs_buffer[1]
        return frame, total_reward, done, info

class FrameStack(gym.Wrapper):
    """
    Description:
        Stack k last frames. Returns lazy array, which is much more memory efficient. Allows action
        stacking too via flag.

    Usage:
        - action_stack: if actions should be tracked and stacked too.
        - reset_action: what action to stack on reset
        - k: is how many frames you want to stack.

    Notes:
        - Have this support actions with dimension greater than 1.
        - If you're stacking actions, then k must include the action frames.
        - If you're stacking actions, then k must be even.
    """
    def __init__(self, env, k, action_stack=False, reset_action=None, lazy=True, use_torch=False):
        gym.Wrapper.__init__(self, env)
        self.total_k = k
        if action_stack:
            assert k % 2 == 0, "{} must be even, something went wrong".format(k)
        self.per_stack = k // 2 if action_stack else k
        self.frames = deque([], maxlen=self.per_stack)
        self.actions = deque([], maxlen=self.per_stack) if action_stack else None
        self.reset_action = reset_action
        self.lazy = lazy
        self.use_torch = use_torch

        shp = env.observation_space.shape
        self.action_dtype = env.action_space.dtype
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=((self.total_k, ) + shp),
                                                dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.per_stack):
            self.frames.append(ob)
            if self.actions is not None:
                self.actions.append(np.ones_like(ob, dtype=self.action_dtype) * self.reset_action)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        if self.actions is not None:
            self.actions.append(np.ones_like(ob, dtype=self.action_dtype) * action)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.per_stack
        if self.actions is not None:
            result = [
                item for frame_action in zip(self.frames, self.actions) for item in frame_action
            ]
        else:
            result = list(self.frames)
        if self.lazy:
            return LazyFrames(result, use_torch=self.use_torch)
        elif self.use_torch:
            return torch.stack(result, dim=0)
        else:
            return np.stack(result, axis=0)

class TorchInterface(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        state = self.env.reset()
        return torch.as_tensor(state)

    def step(self, action):
        np_action = np.array(action)
        state, reward, done, info = self.env.step(np_action)
        state = torch.as_tensor(state)
        reward = torch.as_tensor(reward)
        done = torch.as_tensor(done)
        return state, reward, done, info

class LazyFrames(object):
    """
    Description:
        This object ensures that common frames between the observations are only stored once.  It
        exists purely to optimize memory usage which can be huge for DQN's 1M frames replay buffers.
        This object should only be converted to numpy array before being passed to the model.

    Usage:
        Wrap frames with this object.

    Notes:
        - Can be finicky if used without the OpenAI ReplayBuffer
    """
    def __init__(self, frames, use_torch=False):
        self._frames = frames
        self._use_torch = use_torch

    @property
    def shape(self):
        return self._force().shape

    def _force(self):
        if self._use_torch:
            return torch.stack(self._frames, dim=0)
        else:
            return np.stack(self._frames, axis=0)

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            if self._use_torch:
                out = out.to(dtype)
            else:
                out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return self._frames[i]
