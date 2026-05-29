import torch

import numpy as np
import numba as nb

@nb.njit(cache=True)
def _viterbi(log_prob, log_trans_T, log_p_init):
    n_steps, n_states = log_prob.shape

    state = np.empty(n_steps, dtype=np.int16)
    ptr = np.empty((n_steps, n_states), dtype=np.int16)

    prev = np.empty(n_states, dtype=log_prob.dtype)
    curr = np.empty(n_states, dtype=log_prob.dtype)

    for j in range(n_states):
        prev[j] = log_prob[0, j] + log_p_init[j]

    for t in range(1, n_steps):
        for j in range(n_states):
            row = log_trans_T[j]
            best_val = prev[0] + row[0]
            best_idx = 0

            for i in range(1, n_states):
                v = prev[i] + row[i]
                if v > best_val: best_val, best_idx = v, i

            curr[j] = log_prob[t, j] + best_val
            ptr[t, j] = best_idx

        prev, curr = curr, prev

    best_idx = 0
    best_val = prev[0]

    for j in range(1, n_states):
        v = prev[j]
        if v > best_val: best_val, best_idx = v, j

    state[-1] = best_idx
    for t in range(n_steps - 2, -1, -1):
        state[t] = ptr[t + 1, state[t + 1]]

    return state, best_val

def viterbi_np(prob, transition, p_init=None, return_logp=False, eps=1e-7):
    n_states, _ = prob.shape[-2:]

    if transition.shape != (n_states, n_states): raise ValueError
    if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1): raise ValueError
    if np.any(prob < 0) or np.any(prob > 1): raise ValueError

    if p_init is None: p_init = np.full(n_states, 1.0 / n_states, dtype=prob.dtype)
    elif (np.any(p_init < 0) or not np.allclose(p_init.sum(), 1) or p_init.shape != (n_states,)): raise ValueError

    log_trans = np.ascontiguousarray(np.log(transition + eps)).T
    log_prob = np.ascontiguousarray(np.log(prob + eps))
    log_p_init = np.ascontiguousarray(np.log(p_init + eps))

    if log_prob.ndim == 2: states, logp = _viterbi(log_prob.T, log_trans, log_p_init)
    else:
        batch_shape = log_prob.shape[:-2]
        n_steps = log_prob.shape[-1]

        flat = log_prob.reshape(-1, n_states, n_steps)
        states = np.empty((flat.shape[0], n_steps), dtype=np.int16)
        logp = np.empty(flat.shape[0], dtype=log_prob.dtype)

        for i in range(flat.shape[0]):
            s, lp = _viterbi(flat[i].T, log_trans, log_p_init)
            states[i] = s
            logp[i] = lp

        states = states.reshape(*batch_shape, n_steps)
        logp = logp.reshape(batch_shape)

    if return_logp: return states, logp
    return states

def viterbi(prob, transition, p_init=None, return_logp=False, eps=1e-7):
    if not prob.device.type.startswith(("cuda", "cpu")):
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

    if transition.shape != (n_states, n_states): raise ValueError
    if torch.any(transition < 0) or not torch.allclose(transition.sum(dim=-1), torch.tensor(1.0, dtype=transition.dtype, device=transition.device)): raise ValueError
    if torch.any(prob < 0) or torch.any(prob > 1): raise ValueError

    if p_init is None: p_init = torch.full((n_states,), 1.0 / n_states, dtype=prob.dtype, device=prob.device)
    elif torch.any(p_init < 0) or not torch.allclose(p_init.sum(), torch.tensor(1.0, dtype=p_init.dtype, device=p_init.device)) or p_init.shape != (n_states,): raise ValueError

    log_trans = transition.clamp_min(eps).log()
    log_prob = prob.clamp_min(eps).log()
    log_p_init = p_init.clamp_min(eps).log()

    value = torch.zeros((*batch_shape, n_steps, n_states), dtype=prob.dtype, device=prob.device)
    ptr = torch.zeros((*batch_shape, n_steps, n_states), dtype=torch.long, device=prob.device)

    value[..., 0, :] = log_prob[..., :, 0] + log_p_init
    log_trans_expanded = log_trans.unsqueeze(0)
    
    for t in range(1, n_steps):
        max_vals, argmax_indices = (value[..., t - 1, :].unsqueeze(-1) + log_trans_expanded).max(dim=-2)

        ptr[..., t, :] = argmax_indices
        value[..., t, :] = log_prob[..., :, t] + max_vals

    states = torch.zeros((*batch_shape, n_steps), dtype=torch.long, device=prob.device)
    states[..., -1] = value[..., -1, :].argmax(dim=-1)

    for t in range(n_steps - 2, -1, -1):
        next_state = states[..., t + 1].unsqueeze(-1)
        states[..., t] = ptr[..., t + 1, :].gather(dim=-1, index=next_state).squeeze(-1)

    if return_logp:  return states.to(torch.int64), value[..., -1, :].gather(dim=-1, index=states[..., -1].unsqueeze(-1)).squeeze(-1)
    return states.to(torch.int64)