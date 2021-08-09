import numpy as np
import matplotlib.pyplot as plt

# Result:
# As dimensionality increases, pairwise distance between points drawn
# from tanh(normal(0, 1)) seems to increase as ~sqrt(dim) + c

fig, ax = plt.subplots(1, 1, sharex=True)
axes = [ax]
plt.hist
for ax_idx, dim in enumerate([100, 50, 10, 3, 2]):
    N = 1000000
    points = np.tanh(np.random.normal(size=(N, dim)))
    i = np.random.choice(range(N), size=(N,))
    j = np.random.choice(range(N), size=(N,))

    xi = points[i]
    xj = points[j]
    xi.shape
    xj.shape

    d = np.linalg.norm(xi - xj, axis=-1, ord=2)
    axes[0].hist(d, bins=200, label='{}-D'.format(dim))
    print(d.max())
    # axes[0].set_xlabel('{}-D'.format(dim))
axes[0].legend(loc='upper left')
axes[0].set_xlabel('Pairwise distance')
plt.show()
plt.tight_layout()


#%%
y = np.asarray([11.27194881454762, 8.511510681180757, 4.9817514186646035, 3.2767424590093714, 2.7638492360699627])
x = np.asarray([100, 50, 10, 3, 2])
plt.plot(x, y)
plt.plot(x, np.sqrt(x) + 1.7)
plt.xlabel('D')
plt.ylabel('maximum distance')
