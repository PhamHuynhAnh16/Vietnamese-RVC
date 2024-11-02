import torch
import typing as tp

def center_trim(tensor: torch.Tensor, reference: tp.Union[torch.Tensor, int]):
    ref_size: int
    ref_size = reference.size(-1) if isinstance(reference, torch.Tensor) else reference
    delta = tensor.size(-1) - ref_size
    if delta < 0: raise ValueError("tensor phải lớn hơn tham chiếu. " f"Delta là {delta}.")
    if delta: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
    return tensor