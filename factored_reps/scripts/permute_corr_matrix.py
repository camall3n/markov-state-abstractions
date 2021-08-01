import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

from factored_reps.scripts.seriation import compute_serial_matrix

def shuffle_vars(A, seed=None):
    n_vars = len(A)
    indices = np.arange(n_vars)
    np.random.seed(seed)
    np.random.shuffle(indices)
    B = A[indices, :]
    return B

np.random.seed(0)
size = (20, 20)
A = np.random.randn(*size)
A = np.abs(np.corrcoef(A, rowvar=True))**(2 / 3)
A = shuffle_vars(A)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(321)
im1 = ax.imshow(A, cmap='magma', vmin=0, vmax=1)
ax.set_yticks(range(size[0])), ax.set_xticks(range(size[1]))
ax.set_title('A correlation matrix')

row_ind, col_ind = linear_sum_assignment(-A)
B = A[row_ind, :][:, col_ind]

ax = fig.add_subplot(322)
im2 = ax.imshow(B, cmap='magma', vmin=0, vmax=1)
ax.set_yticks(range(size[0])), ax.set_xticks(range(size[1]))
ax.set_xticklabels(col_ind), ax.set_yticklabels(row_ind)
ax.set_title('Hungarian algorithm')

dist_mat = squareform(pdist(A))

for method, sp in zip(['ward', 'single', 'average', 'complete'], [323, 324, 325, 326]):
    _, res_order, res_linkage = compute_serial_matrix(dist_mat, method)
    C = A[res_order, :][:, res_order]
    ax = fig.add_subplot(sp)
    im2 = ax.imshow(C, cmap='magma', vmin=0, vmax=1)
    ax.set_yticks(range(size[0])), ax.set_xticks(range(size[1]))
    ax.set_xticklabels(res_order), ax.set_yticklabels(res_order)
    ax.set_title('Hier. Clustering (' + method + ')')
plt.tight_layout()
plt.show(ax)
#%%

ax = sns.clustermap(A, metric='correlation')
