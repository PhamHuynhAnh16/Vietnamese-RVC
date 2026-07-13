import torch

import numpy as np
import numba as nb

@nb.njit(cache=True)
def _viterbi(log_prob, log_trans_T, log_p_init):
    """
    Core Numba-accelerated Viterbi algorithm for a single sequence.

    Args:
        log_prob (np.ndarray): Log emission probabilities of shape (n_steps, n_states).
        log_trans_T (np.ndarray): Transposed log transition matrix of shape (n_states, n_states).
        log_p_init (np.ndarray): Log initial state probabilities of shape (n_states,).

    Returns:
        tuple: (state, best_val)
            - state (np.ndarray): Most likely sequence of state indices of shape (n_steps,).
            - best_val (float): The maximum log probability of the best path.
    """

    n_steps, n_states = log_prob.shape
    # Allocate memory for tracking optimal paths and backpointers
    state = np.empty(n_steps, dtype=np.int16)
    ptr = np.empty((n_steps, n_states), dtype=np.int16)

    # Buffers to hold log probabilities for forward dynamic programming
    prev = np.empty(n_states, dtype=log_prob.dtype)
    curr = np.empty(n_states, dtype=log_prob.dtype)

    # Initialization step (t = 0)
    for j in range(n_states):
        prev[j] = log_prob[0, j] + log_p_init[j]

    # Recursion step over time steps
    for t in range(1, n_steps):
        for j in range(n_states):
            row = log_trans_T[j]
            # Find the max log probability coming from the previous states
            best_val = prev[0] + row[0]
            best_idx = 0

            for i in range(1, n_states):
                v = prev[i] + row[i]
                if v > best_val: best_val, best_idx = v, i

            # Store the computed max value and its corresponding backpointer
            curr[j] = log_prob[t, j] + best_val
            ptr[t, j] = best_idx

        # Swap buffers for the next time step iteration
        prev, curr = curr, prev

    # Termination step: find the best final state
    best_idx = 0
    best_val = prev[0]

    for j in range(1, n_states):
        v = prev[j]
        if v > best_val: best_val, best_idx = v, j

    # Backtracking path reconstruction
    state[-1] = best_idx
    for t in range(n_steps - 2, -1, -1):
        state[t] = ptr[t + 1, state[t + 1]]

    return state, best_val

def viterbi_np(prob, transition, p_init=None, return_logp=False, eps=1e-7):
    """
    Computes the most likely sequence of states using the Viterbi algorithm (NumPy version).

    Supports both single sequences and batched inputs.

    Args:
        prob (np.ndarray): Emission probabilities.
        transition (np.ndarray): Transition matrix.
        p_init (np.ndarray, optional): Initial state probabilities. Defaults to uniform distribution.
        return_logp (bool, optional): Whether to return the log probability of the paths. Defaults to False.
        eps (float, optional): Epsilon value to prevent log(0). Defaults to 1e-7.

    Returns:
        np.ndarray or tuple: Best state path sequence, optionally paired with its path log-probability.
            
    Raises:
        ValueError: If array dimensions do not align, or if probability invariants are violated.
    """

    n_states, _ = prob.shape[-2:]

    # Input validation and error raising
    if transition.shape != (n_states, n_states): raise ValueError(f"Transition matrix shape must be {(n_states, n_states)}, got {transition.shape}")
    if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1): raise ValueError("Transition matrix must be non-negative and each row must sum to 1.")
    if np.any(prob < 0) or np.any(prob > 1): raise ValueError("Emission probabilities must be bounded between 0 and 1.")

    # Initialize or validate prior distribution
    if p_init is None: p_init = np.full(n_states, 1.0 / n_states, dtype=prob.dtype)
    elif (np.any(p_init < 0) or not np.allclose(p_init.sum(), 1) or p_init.shape != (n_states,)): raise ValueError(f"p_init must be non-negative, sum to 1, and have shape {(n_states,)}, got shape {p_init.shape if hasattr(p_init, 'shape') else 'unknown'}")

    # Convert to log-space safely using epsilon clipping
    log_trans = np.ascontiguousarray(np.log(transition + eps)).T
    log_prob = np.ascontiguousarray(np.log(prob + eps))
    log_p_init = np.ascontiguousarray(np.log(p_init + eps))

    # Single sequence execution path
    if log_prob.ndim == 2: states, logp = _viterbi(log_prob.T, log_trans, log_p_init)
    else:
        # Batched inputs execution path
        batch_shape = log_prob.shape[:-2]
        n_steps = log_prob.shape[-1]

        # Flatten leading batch dimensions to a single dimension for looping
        flat = log_prob.reshape(-1, n_states, n_steps)
        states = np.empty((flat.shape[0], n_steps), dtype=np.int16)
        logp = np.empty(flat.shape[0], dtype=log_prob.dtype)

        for i in range(flat.shape[0]):
            s, lp = _viterbi(flat[i].T, log_trans, log_p_init)
            states[i] = s
            logp[i] = lp

        # Reshape the output back to match the original batch structure
        states = states.reshape(*batch_shape, n_steps)
        logp = logp.reshape(batch_shape)

    if return_logp: return states, logp
    return states

def viterbi(prob, transition, p_init=None, return_logp=False, eps=1e-7):
    """
    Computes the most likely sequence of states using the Viterbi algorithm (PyTorch version).

    Args:
        prob (torch.Tensor): Emission probabilities.
        transition (torch.Tensor): Transition matrix.
        p_init (torch.Tensor, optional): Initial state probabilities. Defaults to uniform distribution.
        return_logp (bool, optional): Whether to return the log probability of the paths. Defaults to False.
        eps (float, optional): Epsilon value to prevent log(0). Defaults to 1e-7.

    Returns:
        torch.Tensor or tuple: Best state path sequence as torch.int64, optionally paired with its path log-probability tensor.
            
    Raises:
        ValueError: If tensor dimensions do not match, or if probability conditions are violated.
    """

    if not prob.device.type.startswith(("cuda", "cpu")):
        # When tested on other backends, it yielded inaccurate results or failed to work, necessitating a fallback to NumPy.
        states_np, logp_np = viterbi_np(
            prob.detach().cpu().numpy(), 
            transition.detach().cpu().numpy(), 
            p_init.detach().cpu().numpy() if p_init is not None else p_init, 
            return_logp=True,
            eps=eps
        )
        states_pt = torch.as_tensor(states_np, device=prob.device, dtype=torch.int32)

        if return_logp: return states_pt, torch.as_tensor(logp_np, device=prob.device)
        return states_pt

    n_states = prob.shape[-2]
    n_steps = prob.shape[-1]
    batch_shape = prob.shape[:-2]

    # Input validations for tensors
    if transition.shape != (n_states, n_states): raise ValueError(f"Transition matrix shape must be {(n_states, n_states)}, got {transition.shape}")
    if torch.any(transition < 0) or not torch.allclose(transition.sum(dim=-1), torch.tensor(1.0, dtype=transition.dtype, device=transition.device)): raise ValueError("Transition matrix must be non-negative and each row must sum to 1.")
    if torch.any(prob < 0) or torch.any(prob > 1): raise ValueError("Emission probabilities must be bounded between 0 and 1.")

    # Initialize or validate prior distribution
    if p_init is None: p_init = torch.full((n_states,), 1.0 / n_states, dtype=prob.dtype, device=prob.device)
    elif torch.any(p_init < 0) or not torch.allclose(p_init.sum(), torch.tensor(1.0, dtype=p_init.dtype, device=p_init.device)) or p_init.shape != (n_states,): raise ValueError

    # Stable calculation via log-space with clamping minimum boundaries
    log_trans = transition.clamp_min(eps).log()
    log_prob = prob.clamp_min(eps).log()
    log_p_init = p_init.clamp_min(eps).log()

    # Dynamic programming matrices
    value = torch.zeros((*batch_shape, n_steps, n_states), dtype=prob.dtype, device=prob.device)
    ptr = torch.zeros((*batch_shape, n_steps, n_states), dtype=torch.long, device=prob.device)
    # Base initialization step (t = 0)
    value[..., 0, :] = log_prob[..., :, 0] + log_p_init
    log_trans_expanded = log_trans.unsqueeze(0)
    
    # Vectorized forward pass across batched grids
    for t in range(1, n_steps):
        # Adding broadcasted transitions and taking max over previous states
        max_vals, argmax_indices = (value[..., t - 1, :].unsqueeze(-1) + log_trans_expanded).max(dim=-2)

        ptr[..., t, :] = argmax_indices
        value[..., t, :] = log_prob[..., :, t] + max_vals

    # Construct the state trajectories placeholder
    states = torch.zeros((*batch_shape, n_steps), dtype=torch.long, device=prob.device)
    states[..., -1] = value[..., -1, :].argmax(dim=-1)
    # Backward dynamic decoding pass
    for t in range(n_steps - 2, -1, -1):
        states[..., t] = ptr[..., t + 1, :].gather(dim=-1, index=states[..., t + 1].unsqueeze(-1)).squeeze(-1)

    if return_logp:  return states.to(torch.int64), value[..., -1, :].gather(dim=-1, index=states[..., -1].unsqueeze(-1)).squeeze(-1)
    return states.to(torch.int64)