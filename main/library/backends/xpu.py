import torch

from collections import defaultdict
from torch.amp.grad_scaler import _MultiDeviceReplicator, OptState, _refresh_per_optimizer_state

def _amp_update_scale_(input, growth_tracker, found_inf, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
    """
    Updates the scaling factor based on whether infs/NaNs were found in the gradients.

    If infs are found, the scale is multiplied by backoff_factor and the growth tracker resets.
    If no infs are found, the growth tracker increments, and the scale is multiplied by
    growth_factor once the tracker reaches the growth_interval.

    Args:
        input (Tensor): The current scale tensor to be updated in-place.
        growth_tracker (Tensor): Tensor tracking consecutive steps without infs/NaNs.
        found_inf (Tensor): Scalar tensor indicating if infs/NaNs were found (> 0).
        growth_factor (float): Multiplier used to increase the scale. Default: 2.0.
        backoff_factor (float): Multiplier used to decrease the scale. Default: 0.5.
        growth_interval (int): Number of inf-free steps required to grow the scale. Default: 2000.

    Returns:
        Tensor: The updated scale tensor.
    """

    # If infs/NaNs were detected in gradients, decrease the scale immediately
    if found_inf > 0:
        input.mul_(backoff_factor)
        growth_tracker.zero_() # Reset the consecutive successful steps tracker
    else:
        # Increment successful step counter
        growth_tracker.add_(1)
        # Grow the scale factor if interval milestone is reached
        if growth_tracker >= growth_interval:
            input.mul_(growth_factor)
            growth_tracker.zero_() # Reset tracker after growing scale

    return input

def _unscale_grads_(self, optimizer, inv_scale, found_inf, allow_fp16):
    """
    Unscales the gradients of the optimizer's parameters using the inverse scale.

    Also checks for infs/NaNs across multiple devices and data types.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer holding parameters to unscale.
        inv_scale (Tensor): Reciprocal of the current scale factor.
        found_inf (Tensor): Scalar tensor to store whether an inf/NaN was found.
        allow_fp16 (bool): If False, raises an error if float16 gradients are encountered.
    """

    # Replicate scales and inf trackers across target devices automatically
    per_device_inv_scale = _MultiDeviceReplicator(inv_scale)
    per_device_found_inf = _MultiDeviceReplicator(found_inf)

    # Structure to pool gradients dynamically by device and data type
    per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for group in optimizer.param_groups:
            for param in group["params"]:
                if not isinstance(param, torch.Tensor): raise AssertionError(f"expected param to be torch.Tensor, got {type(param).__name__}")
                if param.grad is None: continue
                if (not allow_fp16) and param.grad.dtype == torch.float16: raise ValueError("Attempting to unscale FP16 gradients.")

                # Handle sparse matrix conversions cleanly
                if param.grad.is_sparse:
                    if param.grad.dtype is torch.float16: param.grad = param.grad.coalesce()
                    to_unscale = param.grad._values()
                else: to_unscale = param.grad

                # Bucketize reference tensor configurations
                per_device_and_dtype_grads[to_unscale.device][to_unscale.dtype].append(to_unscale)

        # Batch execute in-place unscaling routine using PyTorch specialized internal C++ extensions
        for device, per_dtype_grads in per_device_and_dtype_grads.items():
            for grads in per_dtype_grads.values():
                torch._amp_foreach_non_finite_check_and_unscale_(grads, per_device_found_inf.get(device), per_device_inv_scale.get(device))

    return per_device_found_inf._per_device_tensors

def unscale_(self, optimizer):
    """
    Divides ("unscales") the optimizer's gradient tensors by the scale factor.

    :meth:`unscale_` is optional, serving cases where you need to
    :ref:`modify or inspect gradients<working-with-unscaled-gradients>`
    between the backward pass(es) and :meth:`step`.
    If :meth:`unscale_` is not called explicitly,  gradients will be unscaled  automatically during :meth:`step`.

    Simple example, using :meth:`unscale_` to enable clipping of unscaled gradients::

        ...
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()

    Args:
        optimizer (torch.optim.Optimizer):  Optimizer that owns the gradients to be unscaled.

    .. note::
        :meth:`unscale_` does not incur a CPU-GPU sync.

    .. warning::
        :meth:`unscale_` should only be called once per optimizer per :meth:`step` call,
        and only after all gradients for that optimizer's assigned parameters have been accumulated.
        Calling :meth:`unscale_` twice for a given optimizer between each :meth:`step` triggers a RuntimeError.

    .. warning::
        :meth:`unscale_` may unscale sparse gradients out of place, replacing the ``.grad`` attribute.
    """

    if not self._enabled: return
    # Check internal scale and growth tracker state validity
    self._check_scale_growth_tracker("unscale_")
    optimizer_state = self._per_optimizer_states[id(optimizer)]

    if optimizer_state["stage"] is OptState.UNSCALED: raise RuntimeError("unscale_() has already been called on this optimizer since the last update().")
    elif optimizer_state["stage"] is OptState.STEPPED: raise RuntimeError("unscale_() is being called after step().")

    if self._scale is None: raise AssertionError("_scale is None in unscale_")

    # Calculate multiplier factor inverses
    inv_scale = self._scale.reciprocal()
    found_inf = torch.full((), 0.0, dtype=torch.float32, device=self._scale.device)

    # Process back-end math executions
    optimizer_state["found_inf_per_device"] = self._unscale_grads_(optimizer, inv_scale, found_inf, False)
    optimizer_state["stage"] = OptState.UNSCALED

def update(self, new_scale = None):
    """
    Update the scale factor.

    If any optimizer steps were skipped the scale is multiplied by ``backoff_factor``
    to reduce it. If ``growth_interval`` unskipped iterations occurred consecutively,
    the scale is multiplied by ``growth_factor`` to increase it.

    Passing ``new_scale`` sets the new scale value manually. (``new_scale`` is not
    used directly, it's used to fill GradScaler's internal scale tensor. So if
    ``new_scale`` was a tensor, later in-place changes to that tensor will not further
    affect the scale GradScaler uses internally.)

    Args:
        new_scale (float or :class:`torch.Tensor`, optional, default=None):  New scale factor.

    .. warning::
        :meth:`update` should only be called at the end of the iteration, after ``scaler.step(optimizer)`` has
        been invoked for all optimizers used this iteration.

    .. warning::
        For performance reasons, we do not check the scale factor value to avoid synchronizations,
        so the scale factor is not guaranteed to be above 1. If the scale falls below 1 and/or
        you are seeing NaNs in your gradients or loss, something is likely wrong. For example,
        bf16-pretrained models are often incompatible with AMP/fp16 due to differing dynamic ranges.
    """

    if not self._enabled: return

    _scale, _growth_tracker = self._check_scale_growth_tracker("update")

    if new_scale is not None:
        if self._scale is None: raise AssertionError("_scale is None in update")

        if isinstance(new_scale, float): self._scale.fill_(new_scale)
        else:
            reason = "new_scale should be a float or a 1-element torch.cuda.FloatTensor or torch.FloatTensor with requires_grad=False."

            if new_scale.device.type != self._device: raise AssertionError(reason)
            if new_scale.numel() != 1: raise AssertionError(reason)
            if new_scale.requires_grad is True: raise AssertionError(reason)

            self._scale.copy_(new_scale)
    else:
        # Pull across distributed device tensors to central hub for examination
        found_infs = [found_inf.to(device=_scale.device, non_blocking=True) for state in self._per_optimizer_states.values() for found_inf in state["found_inf_per_device"].values()]
        if len(found_infs) == 0: raise AssertionError("No inf checks were recorded prior to update.")

        # Combine all sub-tensors together to isolate failures
        found_inf_combined = found_infs[0]
        if len(found_infs) > 1:
            for i in range(1, len(found_infs)):
                found_inf_combined += found_infs[i]

        # Trigger analytical mathematical adjustments
        _amp_update_scale_(_scale, _growth_tracker, found_inf_combined, self._growth_factor, self._backoff_factor, self._growth_interval)

    # Flush current memory maps, refreshing structures ahead of subsequent backprop passes
    self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

def setup_gradscaler():
    """
    Monkey-patches the custom internal scaling methods directly onto the core
    native PyTorch `GradScaler` architecture classes.
    """

    torch.amp.grad_scaler.GradScaler._unscale_grads_ = _unscale_grads_
    torch.amp.grad_scaler.GradScaler.unscale_ = unscale_
    torch.amp.grad_scaler.GradScaler.update = update

def setup_onnxruntime_xpu():
    """
    Attempts to inject OpenVINO Windows path libraries dynamically into the system
    environment, safeguarding against failures via fallback catch blocks.
    """

    try:
        import onnxruntime.tools.add_openvino_win_libs as utils
        utils.add_openvino_libs_to_path()
    except:
        pass