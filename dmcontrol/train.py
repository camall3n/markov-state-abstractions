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
        parser.add_argument('--unique-id', required=True,
                            help='A unique identifier for the experiment')
        args, unknown = parser.parse_known_args()
        other_args = {
            (remove_prefix(key, '--'), val)
            for (key, val) in zip(unknown[::2], unknown[1::2])
        }
        # yapf: enable

        hyperparam_name = '00'
        hyperparams_file = os.path.join('dmcontrol', 'hyperparams', args.alg,
                                        hyperparam_name + '.hyper')
        params = utils_for_q_learning.get_hyper_parameters(hyperparams_file, args.alg)
        params['hyper_parameters_name'] = hyperparam_name
        params['features'] = args.features
        params['alg'] = args.alg
        params['seed_number'] = args.seed

        params['base_dir'] = 'dmcontrol'
        for subdir in ['logs', 'models', 'results', 'hyperparams']:
            subdir_path = os.path.join(params['base_dir'], subdir, args.alg)
            os.makedirs(subdir_path, exist_ok=True)
            params[subdir + '_dir'] = subdir_path

        log_file = os.path.join(params['logs_dir'], '{}.log'.format(args.seed))
        configure_logger(log_file)

        for arg_name, arg_value in other_args:
            if arg_name in params:
                logging.warning("Unknown parameter '{}'".format(arg_value))
            params[arg_name] = arg_value

        utils_for_q_learning.save_hyper_parameters(params, args.unique_id)

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
        self.Q_object.save(tag=self.params['unique_id'],
                           name='model',
                           model_dir=self.params['models_dir'])

if __name__ == '__main__':
    trial = DMControlTrial()
    trial.run()