import os
import sys
import tqdm
import torch
import random
import concurrent.futures

sys.path.append(os.getcwd())

from main.app.variables import config

def center_trim(tensor, reference):
    """
    Trim the input tensor along its last dimension to match the size of a reference.

    Args:
        tensor (torch.Tensor): The tensor to be trimmed.
        reference (Union[torch.Tensor, int]): A reference tensor or an integer 
            representing the target size for the last dimension.

    Returns:
        torch.Tensor: The center-trimmed tensor.

    Raises:
        ValueError: If the tensor size is smaller than the reference size.
    """

    # Get the target size from either the reference tensor's last dimension or directly from an integer
    ref_size = reference.size(-1) if isinstance(reference, torch.Tensor) else reference
    delta = tensor.size(-1) - ref_size
    
    if delta < 0: raise ValueError("Tensor size is smaller than the reference size, cannot trim.")
    
    # If there is a difference, perform symmetric center cropping
    if delta: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]

    return tensor

class DummyPoolExecutor:
    """
    A fallback single-threaded executor mimicking concurrent.futures.ThreadPoolExecutor.
    Useful when running tasks sequentially without multithreading overhead (e.g., on GPU).
    """

    class DummyResult:
        """A wrapper class to simulate a Future object."""

        def __init__(
            self, 
            func, 
            *args, 
            **kwargs
        ):
            self.func = func
            self.args = args
            self.kwargs = kwargs

        def result(self):
            """Execute the function and return the actual result synchronously."""

            return self.func(
                *self.args, 
                **self.kwargs
            )

    def __init__(self, workers=0):
        pass

    def submit(self, func, *args, **kwargs):
        """Submit a task to be wrapped inside DummyResult."""

        return DummyPoolExecutor.DummyResult(func, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return

class TensorChunk:
    """
    Represents a specific chunk/slice of a torch.Tensor without duplicating memory, 
    supporting custom zero-padding mechanisms.
    """

    def __init__(
        self, 
        tensor, 
        offset=0, 
        length=None
    ):
        """
        Initializes the TensorChunk.

        Args:
            tensor (Union[torch.Tensor, TensorChunk]): The source tensor or another TensorChunk.
            offset (int): Starting position inside the source tensor. Defaults to 0.
            length (Optional[int]): Length of the chunk. Defaults to the remaining length.
        """

        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length
        # If length is not specified, take all remaining elements
        length = total_length - offset if length is None else min(total_length - offset, length)

        # Unpack nested TensorChunks to reference the original underlying tensor directly
        if isinstance(tensor, TensorChunk):
            self.tensor = tensor.tensor
            self.offset = offset + tensor.offset
        else:
            self.tensor = tensor
            self.offset = offset

        self.length = length
        self.device = tensor.device

    @property
    def shape(self):
        """Returns the shape of the chunk, overriding the final dimension with the chunk length."""

        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        """
        Extract the chunk and pad it symmetrically with zeros if it exceeds bounds.

        Args:
            target_length (int): The expected output length after padding.

        Returns:
            torch.Tensor: The padded chunk tensor.
        """

        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0
        # Calculate ideal start and end indices for center padding
        start = self.offset - delta // 2
        end = start + target_length
        # Constrain boundaries to avoid IndexError
        correct_start = max(0, start)
        correct_end = min(total_length, end)
        # Calculate required padding for out-of-bound segments
        pad_left = correct_start - start
        pad_right = end - correct_end

        # Slice the real data and apply zero-padding to both ends
        out = torch.nn.functional.pad(
            self.tensor[..., correct_start:correct_end], 
            (pad_left, pad_right)
        )

        assert out.shape[-1] == target_length
        return out

def tensor_chunk(tensor_or_chunk):
    """Ensures the input is wrapped in a TensorChunk object."""

    if isinstance(tensor_or_chunk, TensorChunk): return tensor_or_chunk
    else:
        assert isinstance(tensor_or_chunk, torch.Tensor)
        return TensorChunk(tensor_or_chunk)

def apply_model(
    model, 
    mix, 
    shifts=1, 
    split=True, 
    overlap=0.25, 
    transition_power=1.0, 
    static_shifts=1, 
    set_progress_bar=None, 
    device=None, 
    progress=False, 
    num_workers=0, 
    pool=None,
    hybrid = False,
    weights = None
):
    """
    Apply a separation model to a mixture audio tensor with advanced processing techniques
    including model ensembling (hybrid mode), random time shifts, and overlapped chunk splitting.

    Args:
        model (Any): Single model instance or a list of models (if hybrid=True).
        mix (torch.Tensor): Input audio tensor of shape (batch, channels, length).
        shifts (int): Number of random time shifts to smooth out boundary artifacts.
        split (bool): If True, process audio in overlapping segments.
        overlap (float): Overlap ratio between consecutive chunks [0, 1).
        transition_power (float): Power factor for cross-fading window weights.
        static_shifts (int): Multiplier used solely for tracking progress tracking.
        set_progress_bar (Optional[Callable]): Progress bar update callback.
        device (Optional[Union[str, torch.device]]): Target device (CPU/CUDA).
        progress (bool): If True, display a local tqdm progress bar.
        num_workers (int): Number of worker threads for CPU parallel execution.
        pool (Optional[Any]): Custom thread pool executor.
        hybrid (bool): If True, treats `model` as an ensemble list and blends outputs.
        weights (Optional[List[List[float]]]): Blend weights for each source in hybrid mode.

    Returns:
        torch.Tensor: Separated sources tensor.
    """

    global fut_length, bag_num, prog_bar

    # Setup execution device and fallback worker pool
    device = mix.device if device is None else torch.device(device)
    if pool is None: pool = concurrent.futures.ThreadPoolExecutor(num_workers) if num_workers > 0 and device.type == "cpu" else DummyPoolExecutor()
    # Pack basic configuration parameters for recursive calls
    kwargs = {"shifts": shifts, "split": split, "overlap": overlap, "transition_power": transition_power, "progress": progress, "device": device, "pool": pool, "set_progress_bar": set_progress_bar, "static_shifts": static_shifts}

    # MODE 1: HYybrid Ensemble Processing
    if hybrid:
        estimates, fut_length, prog_bar, current_model = 0, 0, 0, 0
        totals = [0] * len(model[0].sources)
        bag_num = len(model)

        # Iterate over each sub-model in the ensemble
        for sub_model, weight in zip(model, weights):
            sub_model.to(device).eval()
            if config.is_half: sub_model = sub_model.half()

            fut_length += fut_length
            current_model += 1
            # Recursively call apply_model for the single sub-model instance
            out = apply_model(sub_model, mix, **kwargs)

            # Apply weighted combination for each separated stem (source)
            for k, inst_weight in enumerate(weight):
                out[:, k, :, :] *= inst_weight
                totals[k] += inst_weight

            estimates += out
            del out

        # Normalize the accumulated estimates by total assigned weights
        for k in range(estimates.shape[1]):
            estimates[:, k, :, :] /= totals[k]

        return estimates

    assert transition_power >= 1
    batch, channels, length = mix.shape

    # MODE 2: Random Time Shifting (Trick To Improve Quality)
    if shifts:
        kwargs["shifts"] = 0 # Disable nested shift looping in recursion
        max_shift = int(0.5 * model.samplerate)
        padded_mix = tensor_chunk(mix).padded(length + 2 * max_shift)
        out = 0

        # Run inference multiple times with random offsets and average the results
        for _ in range(shifts):
            offset = random.randint(0, max_shift)
            out += apply_model(model, TensorChunk(padded_mix, offset, length + max_shift - offset), **kwargs)[..., max_shift - offset :]

        out /= shifts
        return out
    elif split: # MODE 3: Overlapping Chunk Spitting (For Long Audio)
        kwargs["split"] = False # Avoid redundant splitting in nested calls
        out = torch.zeros(batch, len(model.sources), channels, length, device=mix.device)
        sum_weight = torch.zeros(length, device=mix.device)
        segment = int(model.samplerate * model.segment)

        # Create a linear triangular/trapezoidal fade-in and fade-out cross-fading window
        weight = torch.cat([torch.arange(1, segment // 2 + 1, device=device), torch.arange(segment - segment // 2, 0, -1, device=device)])
        assert len(weight) == segment
        weight = (weight / weight.max()) ** transition_power
        futures = []

        # Submit overlapping audio slices into the execution pool
        for offset in range(0, length, int((1 - overlap) * segment)):
            futures.append((pool.submit(apply_model, model, TensorChunk(mix, offset, segment), **kwargs), offset))
            offset += segment

        if progress: futures = tqdm.tqdm(futures)
        # Gather processed chunks and reconstruct the full signal using overlap-add
        for future, offset in futures:
            if set_progress_bar:
                fut_length = len(futures) * bag_num * static_shifts
                prog_bar += 1
                set_progress_bar(0.1, (0.8 / fut_length * prog_bar))

            chunk_out = future.result()
            chunk_length = chunk_out.shape[-1]

            # Accumulate the weighted chunks into the output buffer
            out[..., offset : offset + segment] += (weight[:chunk_length].to(device) * chunk_out).to(mix.device)
            sum_weight[offset : offset + segment] += weight[:chunk_length].to(mix.device)

        # Normalize by accumulated window weights to prevent volume distortion
        assert sum_weight.min() > 0
        out /= sum_weight
        return out
    else: # MODE 4: Base Model Inference (Direct Inference On Ram Chunk)
        with torch.no_grad():
            # Pad valid length if required by convolutional/architecture constraints
            out = model(tensor_chunk(mix).padded(model.valid_length(length) if hasattr(model, "valid_length") else length).to(device))

        # Trim extra padding introduced by the model architecture back to original length
        return center_trim(out, length)