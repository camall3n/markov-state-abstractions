import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
from sklearn.neighbors import KernelDensity
import sys
import torch

from gridworlds.utils import reset_seeds
import gridworlds.domain.vigorito.vigorito as vigorito

def fit_kde(x, bw=0.03):
    kde = KernelDensity(bandwidth=bw, kernel='tophat')
    kde.fit(x)
    return kde

def fit_ckde(x, c, bw=0.03):
    pc = np.exp(fit_kde(c, bw=bw).score_samples(c))
    kde = KernelDensity(bandwidth=bw)
    kde.fit(x, sample_weight=pc)
    return kde

def MI(x,y):
    xy = np.concatenate([x,y], axis=-1)
    log_pxy = fit_kde(xy).score_samples(xy)
    log_px = fit_kde(x).score_samples(x)
    log_py = fit_kde(y).score_samples(y)
    log_ratio = log_pxy - log_px - log_py
    return np.mean(log_ratio/np.log(2))

def CMI(x,y,c):
    xyc = np.concatenate([x,y,c], axis=-1)
    xc = np.concatenate([x,c], axis=-1)
    yc = np.concatenate([y,c], axis=-1)
    log_pxyc = fit_kde(xyc).score_samples(xyc)
    log_pxc = fit_kde(xc).score_samples(xc)
    log_pyc = fit_kde(yc).score_samples(yc)
    log_pc = fit_kde(c).score_samples(c)
    log_ratio = log_pc + log_pxyc - log_pxc - log_pyc
    return np.mean(log_ratio/np.log(2))

#%%
def get_batch(n):
    s, a = vigorito.run_agent(n_samples=n)
    sp = s[1:,:]
    s = s[:-1,:]
    return s, a, sp

def extract_col(x, col):
    return np.expand_dims(x[:,col],axis=1)

#%%
N = 3000
pred_idx = 0 if len(sys.argv) < 2 else int(sys.argv[1])
threshold = 0.25
s, a, sp = get_batch(N)
f = np.concatenate((s, a), axis=1)
spi = extract_col(sp,pred_idx)
parents = []
print('Finding DBN dependencies for s{}’...'.format(pred_idx))
print()

mi = np.asarray([MI(spi, extract_col(f,i)) for i in range(f.shape[-1])])
mi /= np.sum(mi)
print('Parents(s{}’) ='.format(pred_idx), parents)
print('I(f;s{}’) ~'.format(pred_idx), mi)
idx = np.argmax(mi)
print('Best candidate feature at index {} with value {}.'.format(idx, mi[idx]))
if mi[idx] > threshold:
    print('Adding f{} to parents of s{}’...'.format(idx, pred_idx))
    parents.append(idx)
else:
    print('Value {} not above threshold... stopping.'.format(mi[idx]))
    print()
    print('Parents(s{}’) ='.format(pred_idx), parents)
    sys.exit()
print()

#%%
# Pick s0
s, a, sp = get_batch(N)
f = np.concatenate((s, a), axis=1)
spi = extract_col(sp, pred_idx)
Par_spi = np.concatenate([extract_col(s,i) for i in parents],axis=1)
cmi = np.asarray([CMI(spi, extract_col(f,i), Par_spi) for i in range(f.shape[-1])])
cmi /= np.sum(cmi)
print('Parents(s{}’) ='.format(pred_idx), parents)
print('I(f;s{}’|'.format(pred_idx)+'f'+'f'.join(map(str,parents))+') ~', cmi)
idx = np.argmax(cmi)
print('Best candidate feature at index {} with value {}.'.format(idx, cmi[idx]))
if cmi[idx] > threshold:
    print('Adding f{} to parents of s{}’...'.format(idx, pred_idx))
    parents.append(idx)
else:
    print('Value {} not above threshold... stopping.'.format(cmi[idx]))
    print()
    print('Parents(s{}’) ='.format(pred_idx), parents)
    sys.exit()
print()

#%%
# Pick a0
s, a, sp = get_batch(N)
f = np.concatenate((s, a), axis=1)
spi = extract_col(sp,pred_idx)
Par_spi = np.concatenate([extract_col(f,i) for i in parents], axis=1)
cmi = np.asarray([CMI(spi, extract_col(f,i), Par_spi) for i in range(f.shape[-1])])
cmi /= np.sum(cmi)
print('Parents(s{}’) ='.format(pred_idx), parents)
print('I(f;s{}’|'.format(pred_idx)+'f'+'f'.join(map(str,parents))+') ~', cmi)
print('Best candidate feature at index {} with value {}.'.format(idx, cmi[idx]))
if cmi[idx] > threshold:
    print('Adding f{} to parents of s{}’...'.format(idx, pred_idx))
    parents.append(idx)
else:
    print('Value {} not above threshold... stopping.'.format(cmi[idx]))
print()
print('Parents(s{}’) ='.format(pred_idx), parents)
