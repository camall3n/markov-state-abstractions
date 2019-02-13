import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

N = 1000

def get_sample(N):
    x = np.random.beta(0.1,0.1,N)
    return x

s = np.array(sorted(get_sample(N)))
sns.distplot(s, hist=True, kde=False, bins=50)
plt.show()

#%% Seaborn KDE:
sns.kdeplot(s, bw=0.005, clip=[0,1], shade=True)
plt.show()

#%% Manually implement KDE
def gaussian(x, mu, sigma):
    return 1/np.sqrt(2 * np.pi * sigma**2) * np.exp(-1/(2 * sigma**2) * (x - mu)**2)

def areaplot(x, y, ax=None, label=''):
    if ax is None:
        fig, ax = plt.subplots()
    sns.lineplot(x, y, ax=ax, label=label)
    ax.fill_between(x, 0, y, alpha=0.25)

sigma = 0.002
ds = np.concatenate((np.array((0,)), np.diff(s)))

x = np.linspace(0,1,N)
dx = 1/(N-1)

g = np.array(list(map(lambda mu: gaussian(x, mu, sigma), s)))
p = np.sum(g, axis=0)
p /= np.sum(p * dx)
areaplot(x, p, label='uniform spacing')

g_samp = np.array(list(map(lambda mu: gaussian(s, mu, sigma), s)))
p_samp = np.sum(g_samp, axis=0)
p_samp /= np.sum(p_samp*ds)
areaplot(s, p_samp, ax=plt.gca(), label='sampled spacing')
plt.show()


#%% Estimating entropy manually
h_s = -np.sum(p_samp*np.log(p_samp)*ds)

h_x = -np.sum(p*np.log(p)*dx)

K = lambda x: np.exp(-x**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

# two sample approach
t = get_sample(1000)
st = (s[np.newaxis] - t[:, np.newaxis])
h_st = np.log(len(t)) - 1/len(s) * np.sum(np.log(np.sum(K(st), axis=1)), axis=0)

# one-sample approach
ss = (s[np.newaxis] - s[:, np.newaxis])
h_ss = np.log(len(s)-1) - 1/len(s) * np.sum(np.log(np.sum(K(ss), axis=1)-K(0)), axis=0)

print('h_s =', h_s)
print('h_x =', h_x)
print('h_st =', h_st)
print('h_ss =', h_ss)

#%% Tuning sigma manually
N = 1000
eps = 1e-6
ss = (s[np.newaxis] - s[:, np.newaxis])
sigmas = np.logspace(-4,0,50)
h_ss = np.zeros_like(sigmas)
for i, sigma in enumerate(sigmas):
    K = lambda x: np.exp(-x**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    h_ss[i] = np.log(len(s)-1) - 1/len(s) * np.sum(np.log(eps + np.sum(K(ss), axis=1)-K(0)), axis=0)
fig, ax = plt.subplots()
ax.semilogx(sigmas, h_ss)

best_idx = np.argmin(h_ss)
print('h_hat =', h_ss[best_idx])
print('sigma =', sigmas[best_idx])

#%% Tuning sigma manually
N = 1000
eps = 1e-6
s = np.random.uniform(0,1,N)
ss = (s[np.newaxis] - s[:, np.newaxis])
sigmas = np.logspace(-4,0,50)
h_ss = np.zeros_like(sigmas)
for i, sigma in enumerate(sigmas):
    K = lambda x: np.exp(-x**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    h_ss[i] = np.log(len(s)-1) - 1/len(s) * np.sum(np.log(eps + np.sum(K(ss), axis=1)-K(0)), axis=0)
fig, ax = plt.subplots()
ax.semilogx(sigmas, h_ss)

best_idx = np.argmin(h_ss)
print('h_true =', np.log(1-0))
print('h_hat =', h_ss[best_idx])
print('sigma =', sigmas[best_idx])

#%% Tuning sigma manually
s = np.random.normal(0,1,N)
eps = 1e-6
ss = (s[np.newaxis] - s[:, np.newaxis])
sigmas = np.logspace(-4,1,50)
h_ss = np.zeros_like(sigmas)
for i, sigma in enumerate(sigmas):
    K = lambda x: np.exp(-x**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    h_ss[i] = np.log(len(s)-1) - 1/len(s) * np.sum(np.log(eps + np.sum(K(ss), axis=1)-K(0)), axis=0)
fig, ax = plt.subplots()
ax.semilogx(sigmas, h_ss)

best_idx = np.argmin(h_ss)
print('h_true =', np.log(np.sqrt(2*np.pi*np.e)))
print('h_hat =', h_ss[best_idx])
print('sigma =', sigmas[best_idx])
