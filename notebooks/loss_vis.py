import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker
import seaborn as sns

%matplotlib inline
N = 50
x = np.linspace(0,1,N)
y = np.linspace(0,1,N)
xx = np.tile(x, [N,1])
yy = np.reshape(np.repeat(y, N),[N,-1])

def d_l1(x, y, refx=0.5, refy=0.5):
    return np.abs(x-refx) + np.abs(y-refy)

def d_l2(x, y, refx=0.5, refy=0.5):
    return np.sqrt((x-refx)**2 + (y-refy)**2)

def d_lp(x, y, p, refx=0.5, refy=0.5):
    return (np.abs(x-refx)**p + np.abs(y-refy)**p)**(1/p)

l0 = d_lp(xx, yy, 0.05, refx=0, refy=0)
l1 = d_l1(xx, yy, refx=0, refy=0)
l2 = d_l2(xx, yy, refx=0, refy=0)
lmax = d_lp(xx, yy, 100, refx=0, refy=0)


fig = plt.figure(figsize=(8,5))
ax = fig.subplots(2,2)
ax = ax.flatten()
colors = sns.cubehelix_palette(reverse=True, as_cmap=True)

plots = [l0, l1, l2, lmax]#, l1/(l2+1e-3), l1/(lp+1e-3), l2/(lp+1e-3)]
#'L1/L2', 'L1/Lp', 'L2/Lp'
titles = [r'$L_{1/2}$','$L_1$', '$L_2$', '$L_{max}$']
for i, (d,t) in enumerate(zip(plots, titles)):
    ax[i].set_ylim([1,0])
    ax[i].set_xlim([0,1])
    h = ax[i].contourf(xx, yy, d, cmap=colors)
    # h.contour(N*xx, N*yy, d, cmap='gray')
    ax[i].set_title(t)
    ax[i].invert_yaxis()
    ax[i].axis('square')

plt.tight_layout()
plt.show(fig)
#%%
N = 50
B = 2
x = np.linspace(-B,B,N)
y = np.linspace(-B,B,N)
xx = np.tile(x, [N,1])
yy = np.reshape(np.repeat(y, N),[N,-1])

def d_l1(x, y, refx=0.5, refy=0.5):
    return np.abs(x-refx) + np.abs(y-refy)

def d_l2(x, y, refx=0.5, refy=0.5):
    return np.sqrt((x-refx)**2 + (y-refy)**2)

def d_lp(x, y, p, refx=0.5, refy=0.5):
    return (np.abs(x-refx)**p + np.abs(y-refy)**p)**(1/p)

l0 = d_lp(xx, yy, 0.25, refx=0, refy=0)
l1 = d_l1(xx, yy, refx=0, refy=0)
l2 = d_l2(xx, yy, refx=0, refy=0)
lmax = d_lp(xx, yy, 100, refx=0, refy=0)


fig = plt.figure(figsize=(8,5))
ax = fig.subplots(2,2)
ax = ax.flatten()
colors = sns.cubehelix_palette(reverse=True, as_cmap=True)

plots = [l0/(lmax+1e-3), l1/(l2+1e-3), l1/(lmax+1e-3), l2/(lmax+1e-3)]
#'L1/L2', 'L1/Lp', 'L2/Lp'
titles = [r'$L_0/L_{max}$','$L_1/L_2$', '$L_1/L_{max}$', '$L_2/L_{max}$']
for i, (d,t) in enumerate(zip(plots, titles)):
    ax[i].set_ylim([1,0])
    ax[i].set_xlim([0,1])
    h = ax[i].contourf(xx, yy, d, cmap=colors)
    # h.contour(N*xx, N*yy, d, cmap='gray')
    ax[i].set_title(t)
    ax[i].invert_yaxis()
    ax[i].axis('square')

plt.tight_layout()
plt.show(fig)

#%%
fig = plt.figure(figsize=(8,5))
ax = fig.subplots(1,1)
colors = sns.cubehelix_palette(reverse=True, as_cmap=True)

d = l1/(l2+1e-3)

ax.set_ylim([1,0])
ax.set_xlim([0,1])
h = ax.contourf(xx, yy, d, cmap=colors)
# h.contour(N*xx, N*yy, d, cmap='gray')
ax.set_title(r'$L_2/L_1$')
ax.invert_yaxis()
ax.axis('square')

N = 15
x = np.linspace(0,2,N)
y = np.linspace(0,2,N)
xx = np.tile(x, [N,1])
yy = np.reshape(np.repeat(y, N),[N,-1])
eps = 1e-5
d_dx = yy/(xx**2+eps)
d_dy = -1/(xx+eps)
d_norm = np.sqrt(d_dx**2 + d_dy**2)

d_dx /= (d_norm+eps)
d_dy /= (d_norm+eps)

ax.quiver(xx, yy, d_dx, d_dy, angles='xy', label=r'$-\nabla$ Loss', width=0.003, scale=40)
ax.legend()
ax.set_xlim([0.05,2.0])
plt.show(ax)


plt.show(fig)

#%%
N = 1000
x = np.linspace(0,2,N)
y = np.linspace(0,2,N)
xx = np.tile(x, [N,1])
yy = np.reshape(np.repeat(y, N),[N,-1])
loss = yy/(xx+1e-3) + 1e-5

fig = plt.figure(figsize=(8,5))
ax = fig.subplots(1,1)
cs = ax.contourf(xx, yy, loss, locator=ticker.LogLocator(), cmap=colors)
ax.contourf(xx, yy, loss, locator=ticker.LogLocator(subs='auto'), cmap=colors)
ax.set_xlabel(r"∆z = $||z_{t+1} - z_t||_2$")
ax.set_ylabel(r"Error$(\hat{z}_{t+1})$ = $||\hat{z}_{t+1}-z_{t+1}||_2$")
ax.set_title(r"Loss = Error$(\hat{z}_{t+1})$ / ∆z")
cbar = fig.colorbar(cs)

N = 15
x = np.linspace(0,2,N)
y = np.linspace(0,2,N)
xx = np.tile(x, [N,1])
yy = np.reshape(np.repeat(y, N),[N,-1])
eps = 1e-5
d_dx = yy/(xx**2+eps)
d_dy = -1/(xx+eps)
d_norm = np.sqrt(d_dx**2 + d_dy**2)

d_dx /= (d_norm+eps)
d_dy /= (d_norm+eps)

ax.quiver(xx, yy, d_dx, d_dy, angles='xy', label=r'$-\nabla$ Loss', width=0.003, scale=40)
ax.legend()
ax.set_xlim([0.05,2.0])
plt.show(ax)
