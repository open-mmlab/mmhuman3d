from functools import partial

import torch


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def torch_to_numpy(x):
    assert isinstance(x, torch.Tensor)
    return x.detach().cpu().numpy()
