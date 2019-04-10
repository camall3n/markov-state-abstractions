import json
import matplotlib.pyplot as plt
import numpy as np
import os

tag = 'tag'

log = open(os.path.join('logs',tag,'train.txt'),'r').read().splitlines()
results = [json.loads(item) for item in log]

fig, ax = plt.subplots()
data = zip(*[[item[field] for field in item.keys()] for item in results])
fields = results[0].keys()
results = dict(zip(fields, data))
ax.plot(results['step'], results['L_fwd'])
