"""
References:
    MAE: https://github.com/facebookresearch/mae
"""
import copy
import numpy as np
import torch.distributed as dist


try:
    import torch
    from torch import optim, nn
    from torch.utils import mkldnn as mkldnn_utils
    from . import net_desc as net_desc

    TORCH_ENABLED = True
except:
    TORCH_ENABLED = False


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def random_masking_keep_shape(x, mask_ratio):
    """
    Compared with raw function 'random_masking' in MAE project, here we replace unsample patches with 0 patches rather than delete them.
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    """
    N, L, D = x.shape  # batch, length, dim

    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    len_keep = int(L * (1 - mask_ratio))
    ids_keep = ids_shuffle[:, :len_keep]
    ids_mask = ids_shuffle[:, len_keep:]

    for Ni in range(N):
        x[Ni] = x[Ni].index_fill(0, ids_mask[Ni], 0)

    mask = copy.deepcopy(x)
    for Ni in range(N):
        mask[Ni] = mask[Ni].index_fill(0, ids_keep[Ni], 0)
        mask[Ni] = mask[Ni].index_fill(0, ids_mask[Ni], 1)
    return x, mask


def reshape_data(embeddings, c=1, mode='concat', kys=None):
    embeddings = np.array(embeddings)
    d = embeddings.shape[1]
    if mode == 'mean' or mode == 'add':
        embeddings = embeddings.reshape(-1, c, d).mean(axis=1)
    elif mode == 'chans':
        embeddings = embeddings.reshape(-1, c, d).sum(axis=2)
    else:
        embeddings = embeddings.reshape(-1, c * d)

    ys = []
    if kys is not None:
        for kyi in kys:
            yi = _reshape_data(kyi, c=c)
            ys.append(yi)
        return embeddings, ys
    else:
        return embeddings

def _reshape_data(y, c=1):
    y = np.array(y)
    y = y.reshape(-1, c)

    flag = True
    if c > 1:
        for i in range(1, c):
            flag *= np.any(y[..., 0] == y[..., i])
    assert flag, 'Maybe there are some problems during extract embeddings'

    y = y[..., 0]
    return y

def pop_nan_data(X, kys=None):
    if kys is None:
        X = [Xi for Xi in X if not np.any(np.isnan(Xi))]
        return X
    else:
        data = [(ind, Xi) for (ind, Xi) in enumerate(X) if not np.any(np.isnan(Xi))]
        inds = [datai[0] for datai in data]
        X = [datai[1] for datai in data]
        ys = []
        for kyi in kys:
            yi = None
            if kyi is not None:
                yi = np.array(kyi)[inds]
            ys.append(yi)
        return X, ys

