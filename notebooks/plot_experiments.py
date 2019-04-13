import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from gridworlds.utils import load_experiment

tags = ['exp1', 'exp4', 'exp2', 'exp5']


resultsA = load_experiment('exp1-', coefs={'L_inv': 1.0, 'L_fwd': 0.1, 'L_cpc': 1.0, 'L_fac': 0.1})
resultsB = load_experiment('exp4-', coefs={'L_inv': 1.0, 'L_fwd': 0.1, 'L_cpc': 1.0, 'L_fac': 0.0})

fig, ax = plt.subplots(1, 2, figsize=(8,4), sharey=True)
fig.suptitle('Normalized Mutual Information')
for i, (_, result) in enumerate(resultsA.items()):
    label = 'With $L_F$ regularization' if i==0 else None
    ax[0].plot(result['step'], result['MI'], color='C0', label=label, alpha=0.3)
for i, (_, result) in enumerate(resultsB.items()):
    label = 'Without $L_F$ regularization' if i==0 else None
    ax[1].plot(result['step'], result['MI'], color='C1', label=label, alpha=0.2)
[a.set_xlabel('Updates') for a in ax]
[a.legend() for a in ax]
plt.show()

fig, ax = plt.subplots(1,2,figsize=(8,4), sharey=True)
fig.suptitle('Total Loss vs. Time')
for i, (seed, result) in enumerate(resultsA.items()):
    label = 'With $L_F$ regularization' if i==0 else None
    ax[0].plot(result['step'], result['L'], color='C0', label=label, alpha=0.3)
for i, (seed, result) in enumerate(resultsB.items()):
    label = 'Without $L_F$ regularization' if i==0 else None
    ax[1].plot(result['step'], result['L'], color='C1', label=label, alpha=0.3)
[a.set_xlabel('Updates') for a in ax]
[a.legend() for a in ax]
plt.show()

fig, ax = plt.subplots(1,2,figsize=(8,4), sharey=True)
fig.suptitle('$L_F$ vs. Time')
for i, (seed, result) in enumerate(resultsA.items()):
    label = 'With $L_F$ regularization' if i==0 else None
    ax[0].plot(result['step'], result['L_fac'], color='C0', label=label, alpha=0.3)
for i, (seed, result) in enumerate(resultsB.items()):
    label = 'Without $L_F$ regularization' if i==0 else None
    ax[1].plot(result['step'], result['L_fac'], color='C1', label=label, alpha=0.3)
[a.set_xlabel('Updates') for a in ax]
[a.legend() for a in ax]
plt.show()
