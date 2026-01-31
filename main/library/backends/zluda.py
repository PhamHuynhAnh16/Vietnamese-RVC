import torch

if torch.cuda.is_available() and torch.cuda.get_device_name().endswith("[ZLUDA]"):
    _torch_stft = torch.stft
    _torch_istft = torch.istft

    def z_stft(input, window, *args, **kwargs):
        return _torch_stft(
            input=input.cpu(), window=window.cpu(), *args, **kwargs
        ).to(input.device)
    
    def z_istft(input, window, *args, **kwargs):
        return _torch_istft(
            input=input.cpu(), window=window.cpu(), *args, **kwargs
        ).to(input.device)

    def z_jit(f, *_, **__):
        f.graph = torch._C.Graph()
        return f

    torch.stft = z_stft
    torch.istft = z_istft
    torch.jit.script = z_jit
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)