# Markov State Abstractions

Learning Markov State Abstractions for Deep Reinforcement Learning

https://arxiv.org/abs/2106.04379

An older version of this paper was presented at the 2020 NeurIPS workshop on Deep Reinforcement Learning -- paper available [here](http://irl.cs.brown.edu/pubs/markov_state_abstractions_ws.pdf). Click [here](https://github.com/camall3n/markov-state-abstractions/tree/v2020.12.14-neurips_deep_rl_workshop) for the corresponding code release.

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

- Visual gridworld: `visgrid/`
- DeepMind Control: `dmcontrol/`
