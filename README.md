# Gridworlds

Scalable test domains for quickly running small- to large-scale grid-world experiments.

### Installation

Download the repo and install the dependencies:
```
cd gridworlds
virtualenv env --python=python3
. env/bin/activate
pip install -r requirements.txt
```

### Running
Running one experiment:
```
python -m notebooks.train_rep [args]
```

Running many experiments:
```
./cluster/run.py --command "python -m notebooks.train_rep [cmd args (w/ seed last)]" [cluster args]
```

Rsync updates from grid:
```
rsync -zurP brown:~/path/to/gridworlds/logs .
```
