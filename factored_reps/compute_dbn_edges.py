import numpy as np
import seeding
from sklearn.neighbors import KernelDensity
import sys

from factored_reps import vigorito
import gridworlds.sensors as sensors

def fit_kde(x, bw=0.03):
    kde = KernelDensity(bandwidth=bw)
    kde.fit(x)
    return kde

def MI(x, y, c=None):
    if c is None:
        xy = np.concatenate([x, y], axis=-1)
        log_pxy = fit_kde(xy).score_samples(xy)
        log_px = fit_kde(x).score_samples(x)
        log_py = fit_kde(y).score_samples(y)
        log_ratio = log_pxy - log_px - log_py
    else:
        xyc = np.concatenate([x, y, c], axis=-1)
        xc = np.concatenate([x, c], axis=-1)
        yc = np.concatenate([y, c], axis=-1)
        log_pxyc = fit_kde(xyc).score_samples(xyc)
        log_pxc = fit_kde(xc).score_samples(xc)
        log_pyc = fit_kde(yc).score_samples(yc)
        log_pc = fit_kde(c).score_samples(c)
        log_ratio = log_pc + log_pxyc - log_pxc - log_pyc
    return np.maximum(np.mean(log_ratio / np.log(2)), 0)

def extract_col(x, col):
    return np.expand_dims(x[:, col], axis=1)

def list_cond_vars(feature_indices):
    if feature_indices:
        return '|f' + 'f'.join(map(str, feature_indices))
    else:
        return ''

#%%
N = 2000
pred_idx = 0 if len(sys.argv) < 2 else int(sys.argv[1])
sigma_threshold = 1.65

seeding.seed(0, np)
env = vigorito.VigoritoWorld()
if len(sys.argv) > 2:
    sensor = sensors.SensorChain([
        sensors.PairEntangleSensor(env.n_states, index_a=0, index_b=1),
        # sensors.PairEntangleSensor(env.n_states, index_a=2, index_b=3),
        # sensors.PairEntangleSensor(env.n_states, index_a=0, index_b=2),
    ])
    for s in sensor.sensors:
        print('Entangling f{} and f{}'.format(s.index_a, s.index_b))
    print()
else:
    sensor = sensors.NullSensor()

def get_batch(n):
    s, a = vigorito.run_agent(env, n_samples=n)
    s = sensor.observe(s)
    sp = s[1:, :]
    s = s[:-1, :]
    return s, a, sp

n_features = env.n_states + env.n_actions
remaining_f = list(range(n_features))
parents = []
print('Finding DBN dependencies for s{}’...'.format(pred_idx))
print()

s, a, sp = get_batch(N)
f = np.concatenate((s, a), axis=1)
spi = extract_col(sp, pred_idx)
for i in range(5):
    Par_spi = None if not parents else np.concatenate([extract_col(f, i) for i in parents], axis=1)
    mi = np.asarray([MI(spi, extract_col(f, i), Par_spi) for i in remaining_f])
    print('Parents(s{}’) ='.format(pred_idx), parents)
    print('I(f;s{}’'.format(pred_idx) + list_cond_vars(parents) + ') =')
    print(dict(zip(remaining_f, mi.round(3))))
    n_sigma = (mi - np.mean(mi)) / np.std(mi)
    # print('n_sigma(I) =')
    # print(dict(zip(remaining_f, n_sigma.round(3))))
    idx = np.argmax(mi)
    print('Best candidate feature at index {} with value {} ({} sigma above mean).'.format(
        remaining_f[idx], mi[idx].round(3), n_sigma[idx].round(3)))
    if mi[idx] > 0.01 and n_sigma[idx] > sigma_threshold:
        print('Adding f{} to parents of s{}’...'.format(remaining_f[idx], pred_idx))
        print()
        parents.append(remaining_f[idx])
        remaining_f.remove(remaining_f[idx])
    else:
        print('{} (+{} sigma) not above threshold... stopping.'.format(
            mi[idx].round(3), n_sigma[idx].round(3)))
        break
print()
print('Parents(s{}’) ='.format(pred_idx), parents)
