import os
import torch
import torchaudio

from functools import wraps
from types import SimpleNamespace
from torch.nn import SyncBatchNorm
from hyperpyyaml import load_hyperpyyaml

from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP

MAIN_PROC_ONLY = 0

def fetch(filename, source):
    """
    Resolves an absolute file path given a base directory source and a filename.

    Args:
        filename: Name of the target file.
        source: Base source directory path.

    Returns:
        The absolute path to the designated file.
    """

    return os.path.abspath(os.path.join(source, filename))

def run_on_main(
    func, 
    args=None, 
    kwargs=None, 
    post_func=None, 
    post_args=None, 
    post_kwargs=None, 
    run_post_on_main=False
):
    """
    Executes a function strictly on the main process in DDP setups, synchronizes ranks,
    and handles subsequent post-execution callbacks.

    Args:
        func: Primary function to run on rank 0.
        args: Positional arguments for the primary function.
        kwargs: Keyword arguments for the primary function.
        post_func: Optional callback function to execute after the primary function.
        post_args: Positional arguments for the callback function.
        post_kwargs: Keyword arguments for the callback function.
        run_post_on_main: If True, executes the callback on the main process;
          otherwise executes it on child processes.
    """

    if args is None: args = []
    if kwargs is None: kwargs = {}
    if post_args is None: post_args = []
    if post_kwargs is None: post_kwargs = {}

    # Execute main operation on Rank 0 and hold other processes at the barrier
    main_process_only(func)(*args, **kwargs)
    ddp_barrier()

    # Handle post-execution operations
    if post_func is not None:
        if run_post_on_main: post_func(*post_args, **post_kwargs)
        else:
            # If not explicitly run on main, run on all side-worker processes
            if not if_main_process(): post_func(*post_args, **post_kwargs)
            ddp_barrier()

def is_distributed_initialized():
    """
    Checks if the PyTorch distributed process group is initialized and active.

    Returns:
        True if distributed environment is operational, otherwise False.
    """

    return (
        torch.distributed.is_available() and 
        torch.distributed.is_initialized()
    )

def if_main_process():
    """
    Determines whether the current process is the primary rank (Rank 0).

    Returns:
        True if running on Rank 0 or non-distributed mode, False otherwise.
    """

    if is_distributed_initialized(): return torch.distributed.get_rank() == 0
    else: return True

class MainProcessContext:
    """Context manager that increments the main process execution depth state flag."""

    def __enter__(self):
        global MAIN_PROC_ONLY

        MAIN_PROC_ONLY += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global MAIN_PROC_ONLY

        MAIN_PROC_ONLY -= 1

def main_process_only(function):
    """
    Decorator ensuring that a function is only executed by the main process (rank 0).

    Args:
        function: The function to be restricted to the main process.

    Returns:
        The wrapped process-aware function.
    """

    @wraps(function)
    def main_proc_wrapped_func(*args, **kwargs):
        with MainProcessContext():
            return function(*args, **kwargs) if if_main_process() else None

    return main_proc_wrapped_func

def ddp_barrier():
    """
    Synchronizes all processes across the distributed framework.

    Supports NCCL (CUDA) and XCCL (Intel XPU) backends dynamically.
    """

    # Bypass barrier tracking if within a main-process-only wrapper or not in distributed mode
    if MAIN_PROC_ONLY >= 1 or not is_distributed_initialized(): return

    # Check for hardware backend profile types
    if torch.distributed.get_backend() == torch.distributed.Backend.NCCL: 
        torch.distributed.barrier(
            device_ids=[torch.cuda.current_device()]
        )
    elif hasattr(torch, "xpu") and hasattr(torch.distributed.Backend, "XCCL") and torch.distributed.get_backend() == torch.distributed.Backend.XCCL:
        torch.distributed.barrier(
            device_ids=[torch.xpu.current_device()]
        )
    else: torch.distributed.barrier()

class Resample(torch.nn.Module):
    """
    An audio resampling module wrapping torchaudio.transforms.Resample

    with support for multiple waveform tensor layouts.
    """

    def __init__(
        self, 
        orig_freq=16000, 
        new_freq=16000, 
        *args, 
        **kwargs
    ):
        """
        Initializes the Resample layer.

        Args:
            orig_freq: Initial audio sampling rate frequency.
            new_freq: Target audio sampling rate frequency.
        """

        super().__init__()

        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=orig_freq, 
            new_freq=new_freq, 
            *args, 
            **kwargs
        )

    def forward(self, waveforms):
        """
        Resamples the input waveform tensor to the target sample rate.

        Args:
            waveforms: Input tensor of shape (batch, time) or (batch, time,
              channels).

        Returns:
            Resampled waveform tensor matching the expected input layout format.
        """

        if self.orig_freq == self.new_freq: return waveforms

        unsqueezed = False
        # Layout tracking and shape normalization
        if len(waveforms.shape) == 2:
            waveforms = waveforms.unsqueeze(1)
            unsqueezed = True
        elif len(waveforms.shape) == 3: waveforms = waveforms.transpose(1, 2)
        else: raise ValueError("Unsupported tensor shape. Expected 2D or 3D tensor input layout.")

        # Move the resampler to the current execution target device
        self.resampler.to(waveforms.device) 
        resampled_waveform = self.resampler(waveforms)
        
        # Revert tensor to original shape configuration layout
        return resampled_waveform.squeeze(1) if unsqueezed else resampled_waveform.transpose(1, 2)

class AudioNormalizer:
    """Normalizes, resamples, and conditions audio channels into consistent format standards."""

    def __init__(
        self, 
        sample_rate=16000, 
        mix="avg-to-mono"
    ):
        """
        Initializes the AudioNormalizer.

        Args:
            sample_rate: Target uniform sampling rate frequency.
            mix: Mixing format strategy ('avg-to-mono' or 'keep').
        """

        self.sample_rate = sample_rate

        if mix not in ["avg-to-mono", "keep"]: raise ValueError("Mix mode strategy must be 'avg-to-mono' or 'keep'")

        self.mix = mix
        self._cached_resamplers = {}

    def __call__(self, audio, sample_rate):
        """
        Applies dynamic resampling and track configuration down-mixing on an audio tensor.

        Args:
            audio: Input audio tensor.
            sample_rate: The native sampling rate of the input audio.

        Returns:
            Processed and normalized audio tensor.
        """

        # Instantiate and cache new resamplers dynamically as needed
        if sample_rate not in self._cached_resamplers: 
            self._cached_resamplers[
                sample_rate
            ] = Resample(
                sample_rate, 
                self.sample_rate
            )

        return self._mix(
            # Resample requires a batch dimension; add it, process, then remove it
            self._cached_resamplers[sample_rate](audio.unsqueeze(0)).squeeze(0)
        )

    def _mix(self, audio):
        """Internal down-mixing function based on the selected mix strategy."""

        flat_input = audio.dim() == 1

        if self.mix == "avg-to-mono":
            if flat_input: return audio
            # Average multi-channel stream data across channel axis dimension index 1
            return audio.mean(1)
        
        if self.mix == "keep": return audio

class Pretrained(torch.nn.Module):
    """
    Base class for managing pretrained speech/audio neural network models.
    Handles hyperparameter parsing, compilation (JIT/Inductor), and parallel wrappers.
    """

    HPARAMS_NEEDED, MODULES_NEEDED = [], []

    def __init__(
        self, 
        modules=None, 
        hparams=None, 
        run_opts=None, 
        freeze_params=True
    ):
        """
        Initializes structural base variables and model wrapper attributes.

        Args:
            modules: Dictionary of torch.nn.Module components.
            hparams: Pre-loaded dictionary of configurations and hyperparameters.
            run_opts: Runtime adjustments mapping device and compile overrides.
            freeze_params: If True, locks network parameters into evaluation modes.
        """

        super().__init__()
        # Resolve system parameters mapping defaults configurations profile parameters
        for arg, default in {
            "device": "cpu", 
            "data_parallel_count": -1, 
            "data_parallel_backend": False, 
            "distributed_launch": False, 
            "distributed_backend": "nccl", 
            "jit": False, 
            "jit_module_keys": None, 
            "compile": False, 
            "compile_module_keys": None, 
            "compile_mode": "reduce-overhead", 
            "compile_using_fullgraph": False, 
            "compile_using_dynamic_shape_tracing": False
        }.items():
            if run_opts is not None and arg in run_opts: setattr(self, arg, run_opts[arg])
            elif hparams is not None and arg in hparams: setattr(self, arg, hparams[arg])
            else: setattr(self, arg, default)

        # Store submodule network branches within an accessible ModuleDict container
        self.mods = torch.nn.ModuleDict(modules)
        # Relocate modules onto the designated processing device hardware targets
        for module in self.mods.values():
            if module is not None: module.to(self.device)

        # Enforce validation checks on critical configuration structures
        if self.HPARAMS_NEEDED and hparams is None: raise ValueError("Required hyperparameters mapping structure is missing.")

        if hparams is not None:
            for hp in self.HPARAMS_NEEDED:
                if hp not in hparams: raise ValueError(f"Missing required parameter attribute key: {hp}")

            self.hparams = SimpleNamespace(**hparams)

        # Prepare code layers and initialize normalizing transformations
        self._prepare_modules(freeze_params)
        self.audio_normalizer = hparams.get("audio_normalizer", AudioNormalizer())

    def _prepare_modules(self, freeze_params):
        """Applies optimization compilation routines and registers distribution wrappers."""

        self._compile()
        self._wrap_distributed()

        # Freeze weights and switch modules to evaluation mode if configured
        if freeze_params:
            self.mods.eval()
            for p in self.mods.parameters():
                p.requires_grad = False

    def _compile(self):
        """Compiles modules using torch.compile (PyTorch 2.x Inductor) or torch.jit.script."""

        compile_available = hasattr(torch, "compile")
        if not compile_available and self.compile_module_keys is not None: raise ValueError("torch.compile is not supported on this framework build.")

        # Determine target module targets for compilation
        compile_module_keys = set()
        if self.compile: compile_module_keys = set(self.mods) if self.compile_module_keys is None else set(self.compile_module_keys)

        jit_module_keys = set()
        if self.jit: jit_module_keys = set(self.mods) if self.jit_module_keys is None else set(self.jit_module_keys)

        # Verify that all target keys exist in the module structure
        for name in compile_module_keys | jit_module_keys:
            if name not in self.mods: raise ValueError(f"Module target key '{name}' not found.")

        # Apply torch.compile to the specified modules
        for name in compile_module_keys:
            try:
                module = torch.compile(
                    self.mods[name], 
                    mode=self.compile_mode, 
                    fullgraph=self.compile_using_fullgraph, 
                    dynamic=self.compile_using_dynamic_shape_tracing
                )
            except Exception:
                # Fall back if an unexpected error occurs during structural trace conversion
                continue

            self.mods[name] = module.to(self.device)
            jit_module_keys.discard(name) # Avoid redundant JIT compilation

        # Apply torch.jit.script fallback optimization to remaining keys
        for name in jit_module_keys:
            module = torch.jit.script(self.mods[name])
            self.mods[name] = module.to(self.device)

    def _compile_jit(self):
        """Explicitly invokes the comprehensive module graph compilation workflow."""

        self._compile()

    def _wrap_distributed(self):
        """
        Wraps submodules with DistributedDataParallel (DDP) or DataParallel (DP) wrappers

        based on active distributed cluster launch profiles.
        """

        if not self.distributed_launch and not self.data_parallel_backend: return
        elif self.distributed_launch:
            # Wrap with DistributedDataParallel (DDP) for multi-node/multi-GPU training
            for name, module in self.mods.items():
                if any(p.requires_grad for p in module.parameters()): 
                    self.mods[name] = DDP(
                        SyncBatchNorm.convert_sync_batchnorm(module), 
                        device_ids=[self.device]
                    )
        else:
            # Wrap with standard single-node standard multi-GPU DataParallel (DP)
            for name, module in self.mods.items():
                if any(p.requires_grad for p in module.parameters()): 
                    self.mods[name] = (
                        DP(module)
                    ) if self.data_parallel_count == -1 else (
                        DP(
                            module, 
                            [i for i in range(self.data_parallel_count)]
                        )
                    )

    @classmethod
    def from_hparams(
        cls, 
        source, 
        hparams_file="hyperparams.yaml", 
        overrides={}, 
        download_only=False, 
        overrides_must_match=True, 
        **kwargs
    ):
        """
        Factory method to instantiate the pretrained network class components directly
        from an external HyperPyYAML configuration layout structure.

        Args:
            source: Root target path containing configuration settings files.
            hparams_file: File name of the target configuration file.
            overrides: Dynamic run runtime override adjustments dictionary.
            download_only: Toggles checking asset collection setups exclusively.
            overrides_must_match: If True, enforces override key presence validation.
        """

        # Load yaml definitions parsing custom class instances descriptors
        with open(fetch(filename=hparams_file, source=source)) as fin:
            hparams = load_hyperpyyaml(
                fin, 
                overrides, 
                overrides_must_match=overrides_must_match
            )

        pretrainer = hparams.get("pretrainer", None)
        # Manage external assets if pretrainer class extensions exist
        if pretrainer is not None:
            run_on_main(pretrainer.collect_files, kwargs={"default_source": source})
            if not download_only:
                pretrainer.load_collected()
                return cls(hparams["modules"], hparams, **kwargs)
        else: return cls(hparams["modules"], hparams, **kwargs)

class EncoderClassifier(Pretrained):
    """
    An end-to-end network pipeline designed to map raw vocal input wave vectors

    into lower-dimension speaker embedding clusters and classify labels.
    """

    MODULES_NEEDED = [
        "compute_features", 
        "mean_var_norm", 
        "embedding_model", 
        "classifier"
    ]

    def encode_batch(self, wavs, wav_lens=None, normalize=False):
        """
        Extracts latent speaker embeddings from a batch of raw waveforms.

        Args:
            wavs: Input audio tensor waveform data of shape (batch, time) or (time).
            wav_lens: Normalized relative duration values for each audio sample.
            normalize: Toggles post-extraction normalization via hyperparameter targets.

        Returns:
            Extracted latent embedding tensor vectors.
        """

        # Ensure input tensor matches the required 2D batch dimension layout format
        if len(wavs.shape) == 1: wavs = wavs.unsqueeze(0)
        if wav_lens is None: wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Relocate tensors to device and cast to standard precision float format
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        embeddings = self.mods.embedding_model(
            self.mods.mean_var_norm(
                self.mods.compute_features(wavs), 
                wav_lens
            ), 
            wav_lens
        )

        if normalize: 
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings, 
                torch.ones(embeddings.shape[0], device=self.device)
            )

        return embeddings

    def classify_batch(self, wavs, wav_lens=None):
        """
        Processes audio inputs to extract class probabilities, confidence scores, and labels.

        Args:
            wavs: Raw waveform input data tensors.
            wav_lens: Relative time dimension trackers.

        Returns:
            A tuple containing:
                - out_prob: Prediction probability matrices.
                - score: Highest matched confidence accuracy metrics.
                - index: Raw argmax network predicted index coordinates.
                - decoded: String representations of the target output labels.
        """

        out_prob = self.mods.classifier(self.encode_batch(wavs, wav_lens)).squeeze(1)
        # Extract argmax confidence scores along the final classification target axis
        score, index = out_prob.max(dim=-1)

        # Map index coordinates back to string labels via the label encoder
        return out_prob, score, index, self.hparams.label_encoder.decode_torch(index)

    def forward(self, wavs, wav_lens=None):
        """
        Standard torch forward pass method redirection forwarding arguments directly

        to the batch classification pipeline.
        """

        return self.classify_batch(wavs, wav_lens)