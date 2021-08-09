import torch

from gridworld.models.nnutils import extract

#%%
src = torch.arange(24, dtype=torch.float32).view(3, 4, 2)
idx = torch.tensor([1, 3, 0], dtype=torch.int64)
counter = torch.arange(src.shape[0], dtype=torch.int64)
x = extract(src, idx, idx_dim=1, batch_dim=0)
assert (src[counter, idx] == x).all()

#%%
src = torch.arange(24, dtype=torch.float32).view(3, 4, 2)
idx = torch.tensor([1, 2, 0, 2], dtype=torch.int64)
counter = torch.arange(src.shape[1], dtype=torch.int64)
x = extract(src, idx, idx_dim=0, batch_dim=1)
assert (src[idx, counter] == x).all()

#%%
src = torch.arange(24, dtype=torch.float32).view(3, 4, 2)
idx = torch.tensor([1, 0, 1], dtype=torch.int64)
counter = torch.arange(src.shape[0], dtype=torch.int64)
x = extract(src, idx, idx_dim=2, batch_dim=0)
assert (src[counter, :, idx] == x).all()

#%%
src = torch.arange(24, dtype=torch.float32).view(3, 4, 2)
idx = torch.tensor([1, 0, 1, 1], dtype=torch.int64)
counter = torch.arange(src.shape[1], dtype=torch.int64)
x = extract(src, idx, idx_dim=2, batch_dim=1)
assert (src[:, counter, idx] == x).all()

#%%
src = torch.arange(24, dtype=torch.float32).view(3, 4, 2)
idx = torch.tensor([1, 0, 1], dtype=torch.int64)
try:
    extract(src, idx, idx_dim=0)
except RuntimeError:
    err = True
else:
    err = False
finally:
    assert err

#%%
src = torch.arange(12, dtype=torch.float32).view(3, 4)
idx = torch.tensor([1, 0, 3], dtype=torch.int64)
x = extract(src, idx, idx_dim=1)
assert (x == torch.tensor([1., 4., 11.])).all()

#%%
src = torch.arange(24, dtype=torch.float32).view(3, 4, 2)
idx = torch.tensor([1, 0, 1, 1], dtype=torch.int64)
try:
    extract(src, idx, idx_dim=1, batch_dim=2)
except RuntimeError:
    err = True
else:
    err = False
finally:
    assert err
