# Markov State Abstractions

Image-based gridworld experiment for learning Markov state abstractions.

Presented at the 2020 NeurIPS workshop on Deep Reinforcement Learning -- paper available [here](http://irl.cs.brown.edu/pubs/markov_state_abstractions_ws.pdf).

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
