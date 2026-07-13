import torch

def mean(signals, win_length=9):
    """
    Computes the moving average of a 2D tensor signal while safely handling NaN values.

    This function applies a 1D convolution acts as a uniform moving average filter.
    It masks out `NaN` values dynamically so they do not contaminate the windowed average.

    Args:
        signals (torch.Tensor): Input 2D tensor of shape (batch_size, signal_length).
        win_length (int, optional): Size of the sliding window. Defaults to 9.

    Returns:
        torch.Tensor: The mean-pooled signal tensor with the original shape (batch_size, signal_length).
    """

    # Verify input shape constraint
    assert signals.dim() == 2

    # Add a channel dimension for compatibility with conv1d
    signals = signals.unsqueeze(1)
    # Create a boolean mask where True indicates valid (non-NaN) data
    mask = ~torch.isnan(signals)
    padding = win_length // 2

    # Initialize a uniform 1D convolution kernel filled with ones
    ones_kernel = torch.ones(signals.size(1), 1, win_length, device=signals.device)
    # Calculate the moving sum of valid elements and divide by the count of valid elements per window
    avg_pooled = torch.nn.functional.conv1d(
        torch.where(
            mask, 
            signals, 
            torch.zeros_like(signals) # Replace NaN with 0 temporarily so they don't corrupt the sum
        ), 
        ones_kernel, 
        stride=1, 
        padding=padding
    ) / torch.nn.functional.conv1d(
        mask.float(), # Convert boolean mask to float (1.0 for valid, 0.0 for NaN) to count valid items
        ones_kernel, 
        stride=1, 
        padding=padding
    ).clamp(min=1) # Prevent division-by-zero on windows entirely populated by NaNs

    # Revert regions with zero sum back to NaN to preserve missing data context
    avg_pooled[avg_pooled == 0] = float("nan")
    # Remove the temporary channel dimension and return
    return avg_pooled.squeeze(1)

def median(signals, win_length):
    """
    Computes the moving median of a 2D tensor signal while safely handling NaN values.

    This function extracts sliding windows using `unfold`, sorts the valid elements
    within each window, and adaptively selects the true median based on the number
    of non-NaN elements present in each specific window.

    Args:
        signals (torch.Tensor): Input 2D tensor of shape (batch_size, signal_length).
        win_length (int): Size of the sliding window.

    Returns:
        torch.Tensor: The median-pooled signal tensor with the original shape (batch_size, signal_length).
    """

    # Verify input shape constraint
    assert signals.dim() == 2

    # Add a channel dimension for window processing: (batch, 1, signal_length)
    signals = signals.unsqueeze(1)
    # Identify non-NaN values
    mask = ~torch.isnan(signals)
    padding = win_length // 2

    # Pad signals using reflection mode to avoid boundary artifacts
    x = torch.nn.functional.pad(
        torch.where(
            mask, 
            signals, 
            torch.zeros_like(signals) # Temporarily neutralize NaNs with zeros before padding
        ), 
        (padding, padding), 
        mode="reflect"
    )

    # Pad mask with zeros at boundaries so padded regions are treated as invalid
    mask = torch.nn.functional.pad(
        mask.float(), 
        (padding, padding), 
        mode="constant", 
        value=0
    )

    # Extract overlapping windows across the sequence length dimension
    x = x.unfold(2, win_length, 1)
    mask = mask.unfold(2, win_length, 1)

    # Flatten the trailing window dimensions to make them a contiguous 1D array per window step
    x = x.contiguous().view(x.size()[:3] + (-1,))
    mask = mask.contiguous().view(mask.size()[:3] + (-1,))

    # Push NaN elements to the end of the window during sorting by replacing them with positive infinity (+inf)
    x_sorted, _ = torch.where(mask.bool(), x.float(), float("inf")).to(x).sort(dim=-1)
    # Dynamically compute the median index based on the actual number of valid elements in each window
    idx = ((mask.sum(dim=-1) - 1) // 2).clamp(min=0).unsqueeze(-1).long()

    # Extract the median element using hardware-compatible gathering strategies
    median_pooled = x_sorted.take_along_dim(idx, dim=-1).squeeze(-1) if x.device.type.startswith(("ocl", "privateuseone")) else x_sorted.gather(-1, idx).squeeze(-1)
    # Restore windows that resulted in +inf (all elements were NaN) back to actual NaNs
    median_pooled[torch.isinf(median_pooled)] = float("nan")

    # Remove the temporary channel dimension and return
    return median_pooled.squeeze(1)