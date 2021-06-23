# Markov State Abstractions

Presented at the 2020 NeurIPS workshop on Deep Reinforcement Learning -- paper available [here](http://irl.cs.brown.edu/pubs/markov_state_abstractions_ws.pdf). This repo currently contains the code for reproducing the image-based gridworld experiment.

An updated preprint is available [here](https://arxiv.org/abs/2106.04379); updated code release forthcoming.

### Installation

Clone the repo and install the dependencies:
```
cd markov-state-abstractions
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running
Running one experiment:
```
python -m notebooks.train_rep [args]
python -m notebooks.train_agent [args]
python -m notebooks.convert
python -m notebooks.plot [args]
```
