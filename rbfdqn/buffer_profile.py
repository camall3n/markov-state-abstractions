import time

import gym
from cpprb import ReplayBuffer, create_env_dict, create_before_add_func

import rbfdqn.buffer_class
import dmcontrol.gym_wrappers as wrap

def profile(buf_obj):
    st = time.time()
    for _ in range(200):
        _ = buf_obj.sample(256)
    en = time.time()
    print((en - st))

# env = gym.make("Pendulum-v0")

env = gym.make('dm2gym:PendulumSwingup-v0', environment_kwargs={'flat_observation': True})
env = wrap.FixedDurationHack(env)
env = wrap.ObservationDictToInfo(env, "observations")
env = wrap.RenderOpenCV(env)
env = wrap.Grayscale(env)
env = wrap.ResizeObservation(env, (84, 84))
env = wrap.MaxAndSkipEnv(env, skip=4, max_pool=False)
env = wrap.FrameStack(env, 3, lazy=False)

env_dict = create_env_dict(env)
before_add = create_before_add_func(env)
print(env_dict)

# print("Deque as buffer")
# buf = buffer_class.buffer_class(max_length=int(5e5),seed_number=0)
# s = env.reset()
# for i in range(int(10e5)):
#     a = env.action_space.sample()
#     ns, r, d, _ = env.step(a)
#     buf.append(s, a, r, d, ns)
#     s = ns
#     if i != 0 and i % 1e5 == 0:
#         print("Buffer size: {}".format(i))
#         profile(buf)

print("CPPRB as buffer")
buf = ReplayBuffer(int(5e5), env_dict)
s = env.reset()
for i in range(int(10e5)):
    a = env.action_space.sample()
    ns, r, d, _ = env.step(a)
    buf.add(**before_add(obs=s, act=a, rew=r, done=d, next_obs=ns))
    s = ns
    if i != 0 and i % 1e5 == 0:
        print("Buffer size: {}".format(i))
        profile(buf)
