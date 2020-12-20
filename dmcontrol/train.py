import argparse
import logging
import os

import gym
import numpy as np
import torch
from tqdm import tqdm

from rbfdqn import utils_for_q_learning
from rbfdqn.rbfdqn import Agent
from . import gym_wrappers as wrap

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def configure_logger(filename):
    logging.getLogger().addHandler(logging.FileHandler(filename, mode='w'))

class DMControlTrial():
    def __init__(self):
        params, env, device = self.parse_args()
        self.params = params
        self.env = env
        self.device = device

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # yapf: disable
        parser.add_argument('--features', type=str, default='expert',
                            choices=['visual', 'expert', 'markov', 'markov_smooth'],
                            help='Which type of input features to use')
        parser.add_argument('--alg', type=str, default='rbfdqn',
                            help='Algorithm name')
        parser.add_argument('--seed', '-s', type=int, default=0,
                            help='Random seed')
        parser.add_argument('--agent_tag', required=True, type=str,
                            help='A unique identifier for the agent')
        parser.add_argument('--experiment_name', type=str, default='representation_gap',
                            help='A name for the experiment')
        parser.add_argument('--test', action='store_true',
                            help='Runs a very short-duration experiment to test installation.')
        args, unknown = parser.parse_known_args()
        other_args = {
            (remove_prefix(key, '--'), val)
            for (key, val) in zip(unknown[::2], unknown[1::2])
        }
        # yapf: enable

        if args.test:
            hyperparam_name = '01'
        else:
            hyperparam_name = '00'
        hyperparams_file = os.path.join('dmcontrol', 'hyperparams', args.alg,
                                        hyperparam_name + '.hyper')
        params = utils_for_q_learning.get_hyper_parameters(hyperparams_file, args.alg)
        params['hyper_parameters_name'] = hyperparam_name
        params['features'] = args.features
        params['alg'] = args.alg
        params['seed_number'] = args.seed

        for arg_name, arg_value in other_args:
            if arg_name in params:
                raise KeyError("Unknown parameter '{}'".format(arg_value))
            params[arg_name] = arg_value

        results_dir = os.path.join(
            'dmcontrol',
            'experiments',
            params['env_name'].replace(':', '-').replace('/', '-'),
            args.experiment_name,
            args.agent_tag,
            'seed_{:03d}'.format(args.seed),
        )
        os.makedirs(results_dir, exist_ok=True)
        utils_for_q_learning.save_hyper_parameters(params, results_dir)

        params['results_dir'] = results_dir
        for subdir in ['models']:
            subdir_path = os.path.join(results_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            params[subdir + '_dir'] = subdir_path

        log_file = os.path.join(results_dir, 'log.txt')
        configure_logger(log_file)

        device = utils_for_q_learning.get_torch_device()

        env = gym.make(params['env_name'], environment_kwargs={'flat_observation': True})
        return params, env, device

    def setup(self):
        self.env = wrap.FixedDurationHack(self.env)
        self.env = wrap.ObservationDictToInfo(self.env, "observations")

        feature_type = self.params['features']
        if feature_type == 'expert':
            self.env = wrap.MaxAndSkipEnv(self.env,
                                          skip=self.params['action_repeat'],
                                          max_pool=False)
        else:
            self.env = wrap.RenderOpenCV(self.env)
            self.env = wrap.Grayscale(self.env)
            self.env = wrap.ResizeObservation(self.env, (84, 84))
            self.env = wrap.MaxAndSkipEnv(self.env,
                                          skip=self.params['action_repeat'],
                                          max_pool=False)
            self.env = wrap.FrameStack(self.env, self.params['frame_stack'], lazy=False)
        self.params['env'] = self.env
        self.agent = Agent(self.params, self.env, self.device)

        utils_for_q_learning.set_random_seed(self.params)
        self.returns_list = []
        self.loss_list = []
        self.best_score = -np.Inf

    def teardown(self):
        pass

    def pre_episode(self, episode):
        logging.info("episode {}".format(episode))

    def run_episode(self, episode):
        s, done, t = self.env.reset(), False, 0
        while not done:
            a = self.agent.act(s, episode + 1, 'train')
            sp, r, done, _ = self.env.step(np.array(a))
            t = t + 1
            done_p = False if t == self.env.unwrapped._max_episode_steps else done
            self.agent.buffer_object.append(s, a, r, done_p, sp)
            s = sp

    def post_episode(self, episode):
        logging.debug('episode complete')
        #now update the Q network
        loss = []
        for count in tqdm(range(self.params['updates_per_episode'])):
            temp = self.agent.update()
            loss.append(temp)
        self.loss_list.append(np.mean(loss))

        self.every_n_episodes(self.params['eval_period'], self.evaluate_and_archive, episode)

    def evaluate_and_archive(self, episode, *args):
        episode_scores = []
        for ep in range(self.params['n_eval_episodes']):
            s, G, done, t = self.env.reset(), 0, False, 0
            while done == False:
                a = self.agent.act(s, episode, 'test')
                sp, r, done, _ = self.env.step(np.array(a))
                s, G, t = sp, G + r, t + 1
            episode_scores.append(G)
        avg_episode_score = np.mean(episode_scores)
        logging.info("after {} episodes, learned policy achieved {} average score".format(
            episode, avg_episode_score))
        self.returns_list.append(avg_episode_score)
        utils_for_q_learning.save(self.params['results_dir'], self.returns_list, self.loss_list,
                                  self.params)
        is_best = (avg_episode_score > self.best_score)
        if is_best:
            self.best_score = avg_episode_score
        self.agent.save(is_best)

    def every_n_episodes(self, n, callback, episode, *args):
        if (episode % n == 0) or (episode == self.params['max_episode'] - 1):
            callback(episode, *args)

    def run(self):
        self.setup()
        for episode in range(self.params['max_episode']):
            self.pre_episode(episode)
            self.run_episode(episode)
            self.post_episode(episode)
        self.teardown()

if __name__ == '__main__':
    trial = DMControlTrial()
    trial.run()
