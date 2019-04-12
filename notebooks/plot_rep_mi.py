import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from gridworlds.utils import load_experiment


results1 = load_experiment('exp1-', coefs={'L_inv': 1.0, 'L_fwd': 0.1, 'L_cpc': 1.0, 'L_fac': 0.1})
results4 = load_experiment('exp4-', coefs={'L_inv': 1.0, 'L_fwd': 0.1, 'L_cpc': 1.0, 'L_fac': 0.0})

fig, ax = plt.subplots(1, 2, figsize=(8,4), sharey=True)
for i, (_, result) in enumerate(results1.items()):
    label = 'With $L_F$ regularization' if i==0 else None
    ax[0].plot(result['step'], result['MI'], color='C0', label=label, alpha=0.3)
for i, (_, result) in enumerate(results4.items()):
    label = 'Without $L_F$ regularization' if i==0 else None
    ax[1].plot(result['step'], result['MI'], color='C1', label=label, alpha=0.2)
ax[0].set_xlabel('Updates')
ax[0].set_ylabel('Normalized Mutual Information')
ax[1].set_xlabel('Updates')
ax[0].legend()
ax[1].legend()
plt.show()
