from typing import List, Union

import numpy as np
import torch
from pytorch3d.structures import list_to_padded

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def normalize(value,
              origin_value_range=None,
              out_value_range=(0, 1),
              dtype=None,
              clip=False) -> Union[torch.Tensor, np.ndarray]:
    """Normalize the tensor or array and convert dtype."""
    if origin_value_range is not None:
        value = (value - origin_value_range[0]) / (
            origin_value_range[1] - origin_value_range[0] + 1e-9)

    else:
        value = (value - value.min()) / (value.max() - value.min())
    value = value * (out_value_range[1] -
                     out_value_range[0]) + out_value_range[0]
    if clip:
        value = torch.clip(
            value, min=out_value_range[0], max=out_value_range[1])
    if isinstance(value, torch.Tensor):
        if dtype is not None:
            return value.type(dtype)
        else:
            return value
    elif isinstance(value, np.ndarray):
        if dtype is not None:
            return value.astype(dtype)
        else:
            return value


def tensor2array(image: torch.Tensor) -> np.ndarray:
    """Convert image tensor to array."""
    image = image.detach().cpu().numpy()
    image = normalize(
        image,
        origin_value_range=(0, 1),
        out_value_range=(0, 255),
        dtype=np.uint8)
    return image


def array2tensor(image: np.ndarray) -> torch.Tensor:
    """Convert image array to tensor."""
    image = torch.Tensor(image)
    image = normalize(
        image,
        origin_value_range=(0, 255),
        out_value_range=(0, 1),
        dtype=torch.float32)
    return image


def rgb2bgr(rgbs) -> Union[torch.Tensor, np.ndarray]:
    """Convert color channels."""
    bgrs = [rgbs[..., 2, None], rgbs[..., 1, None], rgbs[..., 0, None]]
    if isinstance(rgbs, torch.Tensor):
        bgrs = torch.cat(bgrs, -1)
    elif isinstance(rgbs, np.ndarray):
        bgrs = np.concatenate(bgrs, -1)
    return bgrs


def align_input_to_padded(tensor=Union[List[torch.Tensor], torch.Tensor],
                          ndim: int = 3,
                          batch_size: int = None,
                          padding_mode: Literal['ones', 'zeros', 'repeat',
                                                'none'] = 'none'):
    if isinstance(tensor, list):
        for i in range(len(tensor)):
            if tensor[i].dim == ndim:
                tensor[i] = tensor[i][0]
        tensor = list_to_padded(tensor, equisized=True)
    assert tensor.ndim in (ndim, ndim - 1)
    if tensor.ndim == ndim - 1:
        tensor = tensor.unsqueeze(0)

    if batch_size is not None:
        current_batch_size = tensor.shape[0]
        if current_batch_size == 1:
            tensor = tensor.repeat_interleave(batch_size, 0)
        elif current_batch_size < batch_size:
            if padding_mode == 'ones':
                tensor = torch.cat([
                    tensor,
                    torch.ones_like(tensor)[:1].repeat_interleave(
                        batch_size - current_batch_size, 0)
                ])
            elif padding_mode == 'ones':
                tensor = torch.cat([
                    tensor,
                    torch.zeros_like(tensor)[:1].repeat_interleave(
                        batch_size - current_batch_size, 0)
                ])
            elif padding_mode == 'repeat':
                tensor = tensor.repeat_interleave(
                    batch_size // current_batch_size + 1, 0)[:batch_size]
            else:
                raise ValueError('Wrong batch_size to allocate,'
                                 ' please specify padding mode.')
        elif current_batch_size > batch_size:
            tensor = tensor[:batch_size]

    return tensor
