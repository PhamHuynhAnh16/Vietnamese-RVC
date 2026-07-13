import torch

def init_zluda():
    """
    Initializes a compatibility environment patch for ZLUDA (CUDA on AMD GPUs).

    This function monkey-patches PyTorch's STFT/iSTFT operations to execute on the CPU 
    as a workaround for hardware backend driver limitations, and configures specific 
    accelerator backend flags for stable execution.
    """

    # Store references to the original native PyTorch STFT and iSTFT operations
    _torch_stft = torch.stft
    _torch_istft = torch.istft

    def z_stft(input, window, *args, **kwargs):
        """
        ZLUDA wrapper for Short-Time Fourier Transform (STFT).
        
        Forces computation onto the CPU to avoid backend driver crashes,
        then projects the final tensor back onto the original target device.
        """

        # Offload tensors to CPU, execute STFT, and cast the output back to the original device
        return _torch_stft(
            input=input.cpu(), window=window.cpu(), *args, **kwargs
        ).to(input.device)
    
    def z_istft(input, window, *args, **kwargs):
        """
        ZLUDA wrapper for Inverse Short-Time Fourier Transform (iSTFT).
        
        Forces computation onto the CPU to avoid backend driver crashes,
        then projects the final tensor back onto the original target device.
        """

        # Offload tensors to CPU, execute iSTFT, and cast the output back to the original device
        return _torch_istft(
            input=input.cpu(), window=window.cpu(), *args, **kwargs
        ).to(input.device)

    def z_jit(f, *_, **__):
        """
        Mock decorator implementation replacing standard JIT compiling scripts.
        """

        f.graph = torch._C.Graph()
        return f

    # Override standard PyTorch namespace hooks with custom ZLUDA-compatible patches
    torch.stft = z_stft
    torch.istft = z_istft
    torch.jit.script = z_jit
    # Disable cuDNN since ZLUDA environments might lack complete deep learning runtime parity
    torch.backends.cudnn.enabled = False
    # Configure Scaled Dot Product Attention (SDPA) math backends
    torch.backends.cuda.enable_math_sdp(True) # Enable stable, high-precision native math fallback path
    torch.backends.cuda.enable_flash_sdp(False) # Disable hardware-specific FlashAttention kernels
    torch.backends.cuda.enable_mem_efficient_sdp(False) # Disable proprietary memory-efficient attention kernels