"""
包含parse_model_string, use_gpu, assign_device, check_mkl, convert_images
References:
    MAE: https://github.com/facebookresearch/mae
"""
import builtins, datetime
import copy
import math
import random
import cv2
import numpy
import numpy as np
import torch.distributed as dist
from pycytominer.operations import Spherize, RobustMAD
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from torch._six import inf
import time
from collections import defaultdict, deque

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
    没有采样的位置用0替换，而不是删除
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence； N:batch数；L：patch的个数；D：patch的边长的平方*通道数
    """
    N, L, D = x.shape  # batch, length, dim

    # sort noise for each sample
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]；每张图片上随机生成一组噪声
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove; 返回燃机索引
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # 为了恢复mask
    len_keep = int(L * (1 - mask_ratio))  # 保留多少个
    ids_keep = ids_shuffle[:, :len_keep]
    ids_mask = ids_shuffle[:, len_keep:]

    # keep the first subset
    for Ni in range(N):
        x[Ni] = x[Ni].index_fill(0, ids_mask[Ni], 0)

    ### generate the binary mask: 0 is keep, 1 is remove 其实cnn和Uformer都没有用到
    # mask = torch.ones([N, L], device=x.device)
    # mask[:, :len_keep] = 0
    # # unshuffle to get the binary mask
    # mask = torch.gather(mask, dim=1, index=ids_restore)  # torch.gather，按索引取一定数量的tensor

    mask = copy.deepcopy(x)  # 用来给cnn中仅计算mask部分的loss的
    for Ni in range(N):  # 0 is keep, 1 is remove， 用于loss计算
        mask[Ni] = mask[Ni].index_fill(0, ids_keep[Ni], 0)
        mask[Ni] = mask[Ni].index_fill(0, ids_mask[Ni], 1)
    return x, mask


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def reshape_data(embeddings, c=1, mode='concat', kys=None):
    embeddings = np.array(embeddings)
    d = embeddings.shape[1]
    if mode == 'mean' or mode == 'add':  # 再传入前，应该做归一化
        embeddings = embeddings.reshape(-1, c, d).mean(axis=1)
    elif mode == 'chans':
        embeddings = embeddings.reshape(-1, c, d).sum(axis=2)  # 如果用平均的话（除以256），就体现不出差异了
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

    flag = True  # 每个通道的inds应该是一样的
    if c > 1:
        for i in range(1, c):
            flag *= numpy.any(y[..., 0] == y[..., i])
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

