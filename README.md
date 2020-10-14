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
python -m notebooks.train_rep [args]
python -m notebooks.train_agent [args]
```

Rsync updates from grid:
```
rsync -zurP brown:~/path/to/gridworlds/logs .
```
