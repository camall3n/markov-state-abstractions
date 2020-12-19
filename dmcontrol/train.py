import argparse
import logging
import os

from rbfdqn.rbfdqn import *
from dmcontrol.gym_wrappers import *

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def configure_logger(filename):
    logging.getLogger().addHandler(logging.FileHandler(filename, mode='w'))

class DMControlTrial(Trial):
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

        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info("Running on the GPU")
        else:
            device = torch.device("cpu")
            logging.info("Running on the CPU")

        env = gym.make(params['env_name'], environment_kwargs={'flat_observation': True})
        return params, env, device

    def encode(self, state):
        if self.encoder is None:
            return state
        return self.encoder(torch.as_tensor(state).float())

    def setup(self):
        self.env = FixedDurationHack(self.env)
        self.env = ObservationDictToInfo(self.env, "observations")

        feature_type = self.params['features']
        if feature_type == 'expert':
            self.env = MaxAndSkipEnv(self.env, skip=self.params['action_repeat'], max_pool=False)
        else:
            self.env = RenderOpenCV(self.env)
            self.env = Grayscale(self.env)
            self.env = ResizeObservation(self.env, (84, 84))
            self.env = MaxAndSkipEnv(self.env, skip=self.params['action_repeat'], max_pool=False)
            self.env = FrameStack(self.env, self.params['frame_stack'], lazy=False)
        self.params['env'] = self.env
        self.agent = Agent(self.params, self.env, self.device)

        super().setup()

    def teardown(self):
        super().teardown()
        self.agent.save()

if __name__ == '__main__':
    trial = DMControlTrial()
    trial.run()