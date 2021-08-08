# Markov State Abstractions

Learning Markov State Abstractions for Deep Reinforcement Learning
https://arxiv.org/abs/2106.04379

### Installation

Download the repo:
```
git clone github.com/camall3n/markov-state-abstractions.git
```

Install the dependencies:
```
cd markov-state-abstractions
git submodule init
git submodule update
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

### Experiments

- Visual gridworld: `markov_abstr/visgrid/`
- DeepMind Control: `markov_abstr/dmcontrol/`
