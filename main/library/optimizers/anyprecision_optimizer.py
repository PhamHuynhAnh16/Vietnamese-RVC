import torch

from torch.optim.optimizer import Optimizer

class AnyPrecisionAdamW(Optimizer):
    """
    Implements AnyPrecisionAdamW optimizer with built-in Kahan Summation.
    
    This optimizer allows training with low-precision states (such as BF16/FP16 
    for momentum, variance, and compensation buffers) to significantly reduce 
    the optimizer's memory footprint, while maintaining high training stability 
    via Kahan Summation accumulation adjustments.
    """

    def __init__(
        self, 
        params, 
        lr=1e-3, 
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=0.0, 
        use_kahan_summation=True, 
        momentum_dtype=torch.bfloat16, 
        variance_dtype=torch.bfloat16, 
        compensation_buffer_dtype=torch.bfloat16
    ):
        """
        Initializes configuration properties for the AnyPrecisionAdamW optimizer.

        Args:
            params (Any): Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): Learning rate step factor coefficient. Defaults to 1e-3.
            betas (Tuple[float, float]): Coefficients used for computing running averages of gradient and its square. Defaults to (0.9, 0.999).
            eps (float): Term added to the denominator to improve numerical stability. Defaults to 1e-8.
            weight_decay (float): Decoupled weight decay (L2 penalty) coefficient. Defaults to 0.0.
            use_kahan_summation (bool): Enables Kahan summation algorithm to mitigate low-precision rounding errors during weight updates. Defaults to True.
            momentum_dtype (torch.dtype): Data type used for first momentum storage. Defaults to torch.bfloat16.
            variance_dtype (torch.dtype): Data type used for second momentum storage. Defaults to torch.bfloat16.
            compensation_buffer_dtype (torch.dtype): Data type used for Kahan errors tracking. Defaults to torch.bfloat16.
        """

        defaults = dict(
            lr=lr, 
            betas=betas, 
            eps=eps, 
            weight_decay=weight_decay, 
            use_kahan_summation=use_kahan_summation, 
            momentum_dtype=momentum_dtype, 
            variance_dtype=variance_dtype, 
            compensation_buffer_dtype=compensation_buffer_dtype
        )

        super().__init__(
            params, 
            defaults
        )

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization parameter update iteration step.

        Args:
            closure (Callable, optional): A closure that re-evaluates the model 
                and returns the loss metric. Defaults to None.
        """

        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            use_kahan_summation = group["use_kahan_summation"]
            momentum_dtype = group["momentum_dtype"]
            variance_dtype = group["variance_dtype"]
            compensation_buffer_dtype = group["compensation_buffer_dtype"]

            for p in group["params"]:
                if p.grad is None: continue
                if p.grad.is_sparse: raise RuntimeError("AnyPrecisionAdamW does not support sparse gradients.")

                state = self.state[p]
                # Initialize state variables on demand
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    # First momentum tracking tensor in targeted low precision
                    state["exp_avg"] = torch.zeros_like(p, dtype=momentum_dtype)
                    # Second momentum variance tracking tensor in targeted low precision
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=variance_dtype)

                    if use_kahan_summation: 
                        # Residual rounding error compensation buffer tracker
                        state["compensation"] = torch.zeros_like(p, dtype=compensation_buffer_dtype)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                grad = p.grad
                # Perform decoupled AdamW weight decay
                if weight_decay: p.data.mul_(1 - lr * weight_decay)

                # Update biased first momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw momentum
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute statistical bias corrections coefficients
                bias_correction1 = 1 - beta1 ** step
                step_size = lr / bias_correction1

                denom_correction = (1 - beta2**step) ** 0.5
                # Calculate stabilized adaptive denominators
                centered_variance = (exp_avg_sq.sqrt() / denom_correction).add_(eps, alpha=1)

                if use_kahan_summation:
                    compensation = state["compensation"]
                    # Accumulate the calculated delta step tracking update directly onto the compensation buffer
                    compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)
                    # Snapshot current weights state before introducing target adjustments
                    temp_buffer = p.detach().clone()
                    # Apply compensated updates onto core high-precision parameters
                    p.data.add_(compensation)
                    # Extract the residual error caused by low-precision truncation or quantization
                    compensation.add_(temp_buffer.sub_(p.data))
                else: 
                    # Standard weight update step path if Kahan buffer mechanism is disabled
                    p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)