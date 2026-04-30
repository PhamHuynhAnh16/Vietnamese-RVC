import torch

from collections import defaultdict
from torch.amp.grad_scaler import _MultiDeviceReplicator, OptState, _refresh_per_optimizer_state

def _amp_update_scale_(input, growth_tracker, found_inf, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
    if found_inf > 0:
        input.mul_(backoff_factor)
        growth_tracker.zero_()
    else:
        growth_tracker.add_(1)
        
        if growth_tracker >= growth_interval:
            input.mul_(growth_factor)
            growth_tracker.zero_()

    return input

def _unscale_grads_(self, optimizer, inv_scale, found_inf, allow_fp16):
    per_device_inv_scale = _MultiDeviceReplicator(inv_scale)
    per_device_found_inf = _MultiDeviceReplicator(found_inf)
    per_device_and_dtype_grads: dict[torch.device, dict[torch.dtype, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for group in optimizer.param_groups:
            for param in group["params"]:
                if not isinstance(param, torch.Tensor): raise AssertionError
                if param.grad is None: continue
                if (not allow_fp16) and param.grad.dtype == torch.float16: raise ValueError

                if param.grad.is_sparse:
                    if param.grad.dtype is torch.float16: param.grad = param.grad.coalesce()
                    to_unscale = param.grad._values()
                else: to_unscale = param.grad

                per_device_and_dtype_grads[to_unscale.device][to_unscale.dtype].append(to_unscale)

        for device, per_dtype_grads in per_device_and_dtype_grads.items():
            for grads in per_dtype_grads.values():
                torch._amp_foreach_non_finite_check_and_unscale_(grads, per_device_found_inf.get(device), per_device_inv_scale.get(device))

    return per_device_found_inf._per_device_tensors

def unscale_(self, optimizer):
    if not self._enabled: return

    self._check_scale_growth_tracker("unscale_")
    optimizer_state = self._per_optimizer_states[id(optimizer)]

    if optimizer_state["stage"] is OptState.UNSCALED: raise RuntimeError
    elif optimizer_state["stage"] is OptState.STEPPED: raise RuntimeError

    if self._scale is None: raise AssertionError

    inv_scale = self._scale.reciprocal()
    found_inf = torch.full((), 0.0, dtype=torch.float32, device=self._scale.device)

    optimizer_state["found_inf_per_device"] = self._unscale_grads_(optimizer, inv_scale, found_inf, False)
    optimizer_state["stage"] = OptState.UNSCALED

def update(self, new_scale = None):
    if not self._enabled: return

    _scale, _growth_tracker = self._check_scale_growth_tracker("update")

    if new_scale is not None:
        if self._scale is None: raise AssertionError

        if isinstance(new_scale, float): self._scale.fill_(new_scale)
        else:
            if new_scale.device.type != self._device: raise AssertionError
            if new_scale.numel() != 1: raise AssertionError
            if new_scale.requires_grad is True: raise AssertionError

            self._scale.copy_(new_scale)
    else:
        found_infs = [found_inf.to(device=_scale.device, non_blocking=True) for state in self._per_optimizer_states.values() for found_inf in state["found_inf_per_device"].values()]
        if len(found_infs) == 0: raise AssertionError

        found_inf_combined = found_infs[0]
        if len(found_infs) > 1:
            for i in range(1, len(found_infs)):
                found_inf_combined += found_infs[i]

        _amp_update_scale_(_scale, _growth_tracker, found_inf_combined, self._growth_factor, self._backoff_factor, self._growth_interval)

    self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

def setup_gradscaler():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.amp.grad_scaler.GradScaler._unscale_grads_ = _unscale_grads_
        torch.amp.grad_scaler.GradScaler.unscale_ = unscale_
        torch.amp.grad_scaler.GradScaler.update = update