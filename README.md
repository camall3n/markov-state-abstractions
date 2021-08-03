# Gridworlds

Scalable test domains for quickly running small- to large-scale grid-world experiments.

### Installation

Download the repo and install the dependencies:
```
cd gridworlds
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running
Running one experiment:
```
python -m markov_abstr.visgrid.train_rep [args]
python -m markov_abstr.visgrid.train_agent [args]
```
