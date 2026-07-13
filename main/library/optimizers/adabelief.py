import math
import torch

from torch.optim.optimizer import Optimizer

class AdaBelief(Optimizer):
    """
    Implements the AdaBelief Optimizer with enhanced PyTorch FOREACH performance.
    
    AdaBelief adapts step sizes based on the "belief" in the current gradient direction.
    Instead of scaling by the raw uncentered variance (like Adam), it scales steps by the
    variance of the prediction residual (gradient minus exponential moving average).
    
    Includes built-in support for:
        - Gradient Centralization (GC) to stabilize training.
        - Rectified learning rate scheduler mechanism (RAdam-style) to reduce warmup needs.
    """

    def __init__(
        self,
        params,
        lr = 1e-4,
        betas = (0.8, 0.99),
        eps = 1e-10,
        weight_decay = 0.0,
        use_gc = False,
        rectify = False
    ):
        """
        Initializes configuration properties for the AdaBelief optimizer.

        Args:
            params (Any): Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): Base target learning rate step factor. Defaults to 1e-4.
            betas (Tuple[float, float]): Coefficients used for computing running averages of gradient and its variance. Defaults to (0.8, 0.99).
            eps (float): Term added to the denominator to improve numerical stability. Defaults to 1e-10.
            weight_decay (float): Decoupled weight decay (L2 penalty) coefficient. Defaults to 0.0.
            use_gc (bool): Activates Gradient Centralization for Conv/Linear layers. Defaults to False.
            rectify (bool): Applies RAdam-style momentum rectification. Defaults to False.

        Raises:
            ValueError: If any initialization parameters violate standard baseline ranges.
        """

        if lr <= 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0: raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0: raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            use_gc=use_gc,
            rectify=rectify
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        """Synchronizes internal structural tracking status dictionaries."""

        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("use_gc", False)
            group.setdefault("rectify", False)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization parameter update iteration step.

        Args:
            closure (Callable, optional): A closure that re-evaluates the model and returns the loss metric. Defaults to None.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps, lr, weight_decay, use_gc, rectify = group["eps"], group["lr"], group["weight_decay"], group["use_gc"], group["rectify"]
            # Tracking list parameters used to trigger multi-tensor vectorized foreach actions
            params_with_grad, grads, exp_avgs, exp_avg_vars = [], [], [], []

            for p in group["params"]:
                if p.grad is None: continue

                params_with_grad.append(p)
                grad = p.grad

                # Optional: Apply Gradient Centralization (GC) to weights across non-scalar axes
                if use_gc and grad.dim() > 1 and grad[0].numel() > 1: grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))
                grads.append(grad)

                state = self.state[p]
                # Initialize state variables if they are empty
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of variance deviations
                    state["exp_avg_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state["exp_avg"])
                exp_avg_vars.append(state["exp_avg_var"])

            if not params_with_grad: continue

            # Synchronize parameter step tracking state counters safely
            state = self.state[params_with_grad[0]]
            state["step"] += 1
            step = state["step"]

            for p in params_with_grad[1:]:
                self.state[p]["step"] = step

            # Calculate mathematical statistical tracking bias correction metrics
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            # Apply standard decoupled weight decay parameter updates
            if weight_decay != 0: torch._foreach_mul_(params_with_grad, 1 - lr * weight_decay)

            # Update first momentum term
            torch._foreach_mul_(exp_avgs, beta1)
            torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

            # Core AdaBelief step: Calculate prediction residual
            grad_residuals = torch._foreach_sub(grads, exp_avgs)

            # Update second momentum variance matrix
            torch._foreach_mul_(exp_avg_vars, beta2)
            torch._foreach_addcmul_(exp_avg_vars, grad_residuals, grad_residuals, value=1 - beta2)

            use_variance = True
            r_t = 1.0 # Default scale factor when rectification is inactive
    
            # Optional: Dynamic RAdam-style momentum rectification sequence calculations
            if rectify:
                rho_inf = 2.0 / (1.0 - beta2) - 1.0
                beta2_t = beta2**step
                rho_t = rho_inf - 2.0 * step * beta2_t / (1.0 - beta2_t)
                # Check if the variance tractability is reliable
                use_variance = rho_t >= 5.0
                if use_variance: 
                    # Compute rectification scale modifier
                    r_t = math.sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf) / ((rho_inf - 4) * (rho_inf - 2) * rho_t))

            # Apply final adaptive step modifiers to base network parameters
            if use_variance:
                # Calculate denominators using fast elementwise sqrt operators
                denom = torch._foreach_sqrt(torch._foreach_add(exp_avg_vars, eps))

                torch._foreach_div_(denom, math.sqrt(bias_correction2))
                # Update weights
                torch._foreach_add_(params_with_grad, torch._foreach_div(exp_avgs, denom), alpha=-(lr * r_t / bias_correction1))
            else: 
                # Fallback path if variance tracker isn't stable yet: update via unscaled momentum steps
                torch._foreach_add_(params_with_grad, exp_avgs, alpha=-(lr / bias_correction1))

        return loss