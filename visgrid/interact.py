import random
from visgrid.taxi.taxi import Taxi5x5, BusyTaxi5x5
from visgrid.agents.qlearningagent import SkilledQLearningAgent
from visgrid.taxi.skills import skills5x5, skill_policy
from visgrid.sensors import IdentitySensor

total_timesteps = 50000
episode_timeout = 2000

epsilon = 0.1

random.seed(0)
env = Taxi5x5()

skill_names = list(skills5x5)
skill_fns = [(lambda n: (lambda: skill_policy(env, n)))(n) for n in skill_names]
skills = dict(zip(skill_names, skill_fns))

agent = SkilledQLearningAgent(options=skills, epsilon=epsilon)
sensor = IdentitySensor

timestep = 0
total_reward = 0

state = env.get_state()
observation = sensor.observe(state)
action = agent.act(observation, reward=0)
while timestep < total_timesteps:
    for t in range(episode_timeout):
        state, reward, done = env.step(action)
        observation = sensor.observe(state)
        timestep += 1

        total_reward += reward
        print(total_reward)

        action = agent.act(observation, reward)

        if done or timestep >= total_timesteps:
            break
    env.reset()
    agent.end_of_episode()
