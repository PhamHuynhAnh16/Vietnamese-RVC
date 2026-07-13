import os
import re
import sys
import glob
import torch

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from random import shuffle
from collections import OrderedDict

sys.path.append(os.getcwd())

from main.app.variables import config, translations, logger

MATPLOTLIB_FLAG = False

def optimizer_state_dict_cpu(optimizer):
    """
    Moves all tensors within the optimizer's state dictionary to the CPU.

    Args:
        optimizer (torch.optim.Optimizer): The PyTorch optimizer object.

    Returns:
        dict: A deep copy of the optimizer's state dict with all tensors on CPU.
    """

    import copy

    # Deep copy the state to prevent modifying the original optimizer state
    opt_state = copy.deepcopy(optimizer.state_dict())
    # Iterate through optimizer state values and detach tensors to CPU
    for state in opt_state["state"].values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.detach().cpu()

    return opt_state

def replace_keys_in_dict(d, old_key_part, new_key_part):
    """
    Recursively replaces substrings within the keys of a dictionary or OrderedDict.

    Args:
        d (dict or OrderedDict): The dictionary to process.
        old_key_part (str): The substring to be replaced.
        new_key_part (str): The substring to replace with.

    Returns:
        dict or OrderedDict: A new dictionary with updated keys.
    """

    # Maintain OrderedDict type if the input is an OrderedDict
    updated_dict = OrderedDict() if isinstance(d, OrderedDict) else {}

    for key, value in d.items():
        # Recursively apply to nested dictionaries, otherwise keep the value
        updated_dict[(
            # Replace the substring if the key is a string
            key.replace(old_key_part, new_key_part) if isinstance(key, str) else key
        )] = (
            replace_keys_in_dict(value, old_key_part, new_key_part) if isinstance(value, dict) else value
        )
    
    return updated_dict

def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    """
    Loads a model and optional optimizer state from a checkpoint file.
    Handles legacy weight key conversions (.weight_v / .weight_g) to the newer parametrization format.

    Args:
        checkpoint_path (str): Path to the checkpoint file (.pth).
        model (torch.nn.Module): The model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer instance. Defaults to None.
        load_opt (int, optional): Flag to determine whether to load optimizer state (1 to load). Defaults to 1.

    Returns:
        tuple: (model, optimizer, learning_rate, iteration, scaler_state)
    """

    # Verify checkpoint file existence
    assert os.path.isfile(checkpoint_path), translations["not_found_checkpoint"].format(checkpoint_path=checkpoint_path)

    # Load checkpoint safely and map keys from old formats to PyTorch's parametrization standard
    checkpoint_dict = replace_keys_in_dict(
        replace_keys_in_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True), 
            ".weight_v", 
            ".parametrizations.weight.original1"
        ), 
        ".weight_g", 
        ".parametrizations.weight.original0"
    )

    # Map checkpoint weights into current state dict framework, falling back to current if missing
    new_state_dict = {
        k: checkpoint_dict["model"].get(k, v) 
        for k, v in (model.module.state_dict() if hasattr(model, "module") else model.state_dict()).items()
    }

    # Load state dict non-strictly to tolerate missing/extra keys
    model.module.load_state_dict(new_state_dict, strict=False) if hasattr(model, "module") else model.load_state_dict(new_state_dict, strict=False)
    # Optional loading of the optimizer state
    if optimizer and load_opt == 1: optimizer.load_state_dict(checkpoint_dict.get("optimizer", {}))
    logger.debug(translations["save_checkpoint"].format(checkpoint_path=checkpoint_path, checkpoint_dict=checkpoint_dict['iteration']))

    return (
        model, 
        optimizer, 
        checkpoint_dict.get("learning_rate", 0), 
        checkpoint_dict["iteration"], 
        checkpoint_dict.get("scaler", {})
    )

def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path, scaler):
    """
    Saves the current training states (model, optimizer, iteration, lr, scaler) to a file.
    Converts modern PyTorch parametrization keys back to legacy weight keys for compatibility.

    Args:
        model (torch.nn.Module): The model instance.
        optimizer (torch.optim.Optimizer): The optimizer instance.
        learning_rate (float): Current learning rate.
        iteration (int): Current training iteration/step.
        checkpoint_path (str): Target file path where checkpoint will be saved.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.
    """

    state_dict = (model.module.state_dict() if hasattr(model, "module") else model.state_dict())
    # Move weights/states to CPU if specialized computing devices are used
    if config.device.startswith("privateuseone"):
        model_state = {k: v.detach().cpu() for k, v in state_dict.items()}
        model_optimizer = optimizer_state_dict_cpu(optimizer)
    else:
        model_state = state_dict
        model_optimizer = optimizer.state_dict()

    # Revert newer serialization keys back to legacy format before saving for backward compatibility
    torch.save(
        replace_keys_in_dict(
            replace_keys_in_dict({
                "model": model_state, 
                "iteration": iteration, 
                "optimizer": model_optimizer, 
                "learning_rate": learning_rate, 
                "scaler": scaler.state_dict()
            }, ".parametrizations.weight.original1", ".weight_v"), 
            ".parametrizations.weight.original0", ".weight_g"
        ), 
        checkpoint_path
    )

    logger.info(translations["save_model"].format(checkpoint_path=checkpoint_path, iteration=iteration))

def summarize(
    writer, 
    global_step, 
    scalars={}, 
    histograms={}, 
    images={}, 
    audios={}, 
    audio_sample_rate=22050
):
    """
    Logs multi-modal training metrics and summaries to TensorBoard.

    Args:
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        global_step (int): Current global step in training.
        scalars (dict, optional): Dict of scalar metrics {tag: value}. Defaults to {}.
        histograms (dict, optional): Dict of histogram metrics {tag: values}. Defaults to {}.
        images (dict, optional): Dict of image tensors {tag: image}. Defaults to {}.
        audios (dict, optional): Dict of audio waveforms {tag: audio}. Defaults to {}.
        audio_sample_rate (int, optional): Sampling rate for audio logs. Defaults to 22050.
    """

    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)

    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)

    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")

    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sample_rate)

def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    """
    Finds the latest saved checkpoint in a directory based on numerical sorting of its filename.

    Args:
        dir_path (str): Path to the directory containing checkpoints.
        regex (str, optional): Search wildcard pattern. Defaults to "G_*.pth".

    Returns:
        str or None: Path to the latest checkpoint file, or None if none found.
    """

    checkpoints = sorted(
        glob.glob(
            os.path.join(dir_path, regex)
        ), 
        key=lambda f: int("".join(filter(str.isdigit, f))) # Sort purely by the digits found in the filename
    )
    return checkpoints[-1] if checkpoints else None

def plot_spectrogram_to_numpy(spectrogram):
    """
    Converts a spectrogram tensor into an RGB numpy image array using Matplotlib.

    Args:
        spectrogram (np.ndarray or torch.Tensor): The 2D spectrogram grid.

    Returns:
        np.ndarray: An RGB image representation of the spectrogram with shape (H, W, 3).
    """

    global MATPLOTLIB_FLAG

    # Switch backend to headless 'Agg' to avoid GUI overhead during distributed training
    if not MATPLOTLIB_FLAG:
        plt.switch_backend("Agg")
        MATPLOTLIB_FLAG = True

    fig, ax = plt.subplots(figsize=(10, 2))
    plt.colorbar(ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none"), ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    fig.canvas.draw()
    plt.close(fig)

    # Attempt to extract raw buffer as RGB array; fallback if canvas rendering differs
    try:
        data = np.array(
            fig.canvas.renderer.buffer_rgba(), 
            dtype=np.uint8
        ).reshape(
            fig.canvas.get_width_height()[::-1] + (4,)
        )[:, :, :3]
    except:
        data = np.fromstring(
            fig.canvas.tostring_rgb(), 
            dtype=np.uint8, 
            sep=""
        ).reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )

    return data

def load_wav_to_torch(full_path):
    """
    Loads an audio file from a path into a PyTorch FloatTensor.

    Args:
        full_path (str): Path to the audio file.

    Returns:
        tuple: (torch.FloatTensor of audio data, int sample rate)
    """

    data, sample_rate = sf.read(full_path, dtype=np.float32)
    return torch.FloatTensor(data.astype(np.float32)), sample_rate

def load_filepaths_and_text(filename, split="|"):
    """
    Read a text file containing data paths and convert it into a list.

    Args:
        filename (str): Path to the metadata file.
        split (str, optional): String delimiter. Defaults to "|".

    Returns:
        list of list: A parsed grid containing components of each line.
    """

    with open(filename, encoding="utf-8") as f:
        return [line.strip().split(split) for line in f]
    
class HParams:
    """
    A Hyperparameters configuration container class wrapper that converts nested dictionaries 
    into an object wrapper, enabling attribute-style access (e.g., hps.model.lr).
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            # Recursively turn dictionaries into child HParams objects
            self[k] = HParams(**v) if isinstance(v, dict) else v

    def keys(self):
        return self.__dict__.keys()
    
    def items(self):
        return self.__dict__.items()
    
    def values(self):
        return self.__dict__.values()
    
    def __len__(self):
        return len(self.__dict__)
    
    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__
    
    def __repr__(self):
        return repr(self.__dict__)

def get_device(gpus):
    """
    Resolves the targeted hardware acceleration device context based on config and string requests.

    Args:
        gpus (str): String identifier for allocation (e.g., "0-1-2" or "-").

    Returns:
        tuple: (torch.device object, list of active GPU IDs, int count of devices)
    """

    if gpus == "-":
        device, gpus = torch.device("cpu"), [0]
        n_gpus = 1
        logger.warning(translations["not_gpu"])
    elif config.device.startswith("cuda"):
        device, gpus = torch.device("cuda"), [int(item) for item in gpus.split("-")]
        n_gpus = len(gpus)
    elif config.device.startswith("xpu"):
        device, gpus = torch.device("xpu"), [int(item) for item in gpus.split("-")]
        n_gpus = len(gpus)
    elif config.device.startswith("ocl"):
        device, gpus = torch.device("ocl"), [int(item) for item in gpus.split("-")]
        n_gpus = len(gpus)
    elif config.device.startswith("privateuseone"):
        device, gpus = torch.device("privateuseone"), [int(item) for item in gpus.split("-")]
        n_gpus = len(gpus)
    elif config.device.startswith("mps"):
        device, gpus = torch.device("mps"), [0]
        n_gpus = 1
    else:
        device, gpus = torch.device("cpu"), [0]
        n_gpus = 1
        logger.warning(translations["not_gpu"])

    return device, gpus, n_gpus

def get_optimizer(net_g, net_d, optimizer_choice = "AdamW", learning_rate = 1e-4, betas = [0.8, 0.99], eps = 1e-9, g_lr_coeff = 1.0, d_lr_coeff = 1.0):
    """
    Initializes and selects the optimization engines for both Generator (net_g) and Discriminator (net_d).

    Args:
        net_g (torch.nn.Module): Generator model.
        net_d (torch.nn.Module): Discriminator model.
        optimizer_choice (str, optional): Optimization algorithm name. Defaults to "AdamW".
        learning_rate (float, optional): Base learning rate. Defaults to 1e-4.
        betas (list, optional): Beta coefficients for Adam-based optimizers. Defaults to [0.8, 0.99].
        eps (float, optional): Term added to the denominator to improve numerical stability. Defaults to 1e-9.
        g_lr_coeff (float, optional): Multiplier scaling factor for Generator learning rate. Defaults to 1.0.
        d_lr_coeff (float, optional): Multiplier scaling factor for Discriminator learning rate. Defaults to 1.0.

    Returns:
        tuple: (optim_g, optim_d) instances
    """

    # Dynamic optimizer factory imports based on user choice
    if optimizer_choice == "AnyPrecisionAdamW" and config.brain:
        from main.library.optimizers.anyprecision_optimizer import AnyPrecisionAdamW

        optimizer_optim = AnyPrecisionAdamW
    elif optimizer_choice == "RAdam":
        from torch.optim import RAdam

        optimizer_optim = RAdam
    elif optimizer_choice == "AdaBelief":
        from main.library.optimizers.adabelief import AdaBelief

        optimizer_optim = AdaBelief
    else:
        from torch.optim import AdamW

        optimizer_optim = AdamW
    
    optim_g = optimizer_optim(net_g.parameters(), learning_rate * g_lr_coeff, betas=betas if not optimizer_choice.startswith("AdaBelief") else 1e-8, eps=eps)
    optim_d = optimizer_optim(net_d.parameters(), learning_rate * d_lr_coeff, betas=betas if not optimizer_choice.startswith("AdaBelief") else 1e-8, eps=eps)
    
    return optim_g, optim_d

def get_scheduler(optim_g, optim_d, optimizer_choice = "AdamW", total_epoch = 100, epoch_str = 1, use_cosine_annealing_lr = False, lr_decay = 0.999875):
    """
    Creates learning rate schedules for the Generator and Discriminator optimizers.

    Args:
        optim_g (torch.optim.Optimizer): Optimizer for Generator.
        optim_d (torch.optim.Optimizer): Optimizer for Discriminator.
        optimizer_choice (str, optional): Name of chosen optimizer. Defaults to "AdamW".
        total_epoch (int, optional): Max training epoch target for Cosine Annealing. Defaults to 100.
        epoch_str (int, optional): Starting epoch offset. Defaults to 1.
        use_cosine_annealing_lr (bool, optional): Force cosine schedule. Defaults to False.
        lr_decay (float, optional): Multiplicative decay gamma factor for ExponentialLR. Defaults to 0.999875.

    Returns:
        tuple: (scheduler_g, scheduler_d) Learning rate schedulers.
    """

    # Apply CosineAnnealing fallback conditions, otherwise default to ExponentialLR decay
    if optimizer_choice == "AdaBelief" or use_cosine_annealing_lr:
        scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=total_epoch, eta_min=1e-6, last_epoch=epoch_str - 2)
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, T_max=total_epoch, eta_min=1e-6, last_epoch=epoch_str - 2)
    else:
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=lr_decay, last_epoch=epoch_str - 2)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=lr_decay, last_epoch=epoch_str - 2)
    
    return scheduler_g, scheduler_d

def cleanup_training(experiment_dir):
    """
    Cleans up training checkpoints, index tracks, and evaluation directories inside an experiment directory.

    Args:
        experiment_dir (str): Base root path of the experiment directory.
    """

    for root, dirs, files in os.walk(experiment_dir, topdown=False):
        # Remove target training artifact files
        for name in files:
            if name.endswith((".0", ".pth", ".index")): os.remove(os.path.join(root, name))

        # Empty and strip out structural 'eval' metric logging directories
        for name in dirs:
            if name == "eval":
                folder_path = os.path.join(root, name)

                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    if os.path.isfile(item_path): os.remove(item_path)

                os.rmdir(folder_path)

def get_training_data(info, pitch_guidance):
    """
    Deconstructs a data loader's batch tensor list into cleanly named variables based on pitch features.

    Args:
        info (list): List of tensors yielded by the DataLoader pipeline batch.
        pitch_guidance (bool): Flag indicating if pitch/F0 is used for training.

    Returns:
        tuple: (phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, sid)
    """

    phone, phone_lengths = info[0], info[1]
    # Map indexes conditionally based on whether F0/pitch is part of the batch vectors
    if pitch_guidance:
        pitch, pitchf = info[2], info[3]
        spec, spec_lengths, wave, sid = info[4], info[5], info[6], info[8]
    else:
        pitch = pitchf = None
        spec, spec_lengths, wave, sid = info[2], info[3], info[4], info[6]
    
    return phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, sid

def transform_tensor_into_cache(device, device_id, cache_data_in_gpu, cache, train_loader):
    """
    Caches an entire training data loader directly into a hardware device memory (GPU/XPU/etc.) 
    to maximize iteration speed, or falls back to an standard iterator if caching is disabled.

    Args:
        device (torch.device): Device target context wrapper.
        device_id (int or torch.device): Local computing accelerator rank index.
        cache_data_in_gpu (bool): Flag to trigger total cache storage in device memory.
        cache (list): Existing cache list placeholder matrix.
        train_loader (torch.utils.data.DataLoader): Standard training batch data loader.

    Returns:
        tuple: (data_iterator object, updated cache list structures)
    """

    if device.type == "cuda" and cache_data_in_gpu: # CUDA execution branch caching
        data_iterator = cache

        if cache == []:
            for batch_idx, info in enumerate(train_loader):
                cache.append(
                    (batch_idx, [
                        tensor.cuda(device_id, non_blocking=True) 
                        for tensor in info
                    ])
                )
        else: 
            shuffle(cache)
    elif device.type == "xpu" and cache_data_in_gpu: # Intel XPU execution branch caching
        data_iterator = cache

        if cache == []:
            for batch_idx, info in enumerate(train_loader):
                cache.append(
                    (batch_idx, [
                        tensor.xpu(device_id, non_blocking=True) 
                        for tensor in info
                    ])
                )
        else: 
            shuffle(cache)
    elif device.type in ["privateuseone", "ocl"] and cache_data_in_gpu: # OpenCL & Custom Device (privateuseone) execution branch caching
        data_iterator = cache

        if cache == []:
            for batch_idx, info in enumerate(train_loader):
                cache.append(
                    (batch_idx, [
                        tensor.to(device_id if device.type == "ocl" else device, non_blocking=True) 
                        for tensor in info
                    ])
                )
        else: 
            shuffle(cache)
    else: # Default fallback: streaming directly from CPU/Disk data loader every epoch
        data_iterator = enumerate(train_loader)
    
    return data_iterator, cache

def transforming_computing_devices(info, device, device_id, cache_data_in_gpu):
    """
    Moves an in-flight batch data tensor collection to the target device memory 
    on-the-fly (only if memory-wide caching was not utilized).

    Args:
        info (list): List of tensors from data step.
        device (torch.device): Target hardware device configuration.
        device_id (int or torch.device): Local execution device ID reference.
        cache_data_in_gpu (bool): Flag indicating if dataset was pre-cached.

    Returns:
        list: Moved tensor collections.
    """

    # Process only if tensors are not already cached in GPU memory
    if device.type == "cuda" and not cache_data_in_gpu: 
        info = [
            tensor.cuda(device_id, non_blocking=True) 
            for tensor in info
        ]  
    elif device.type == "xpu" and not cache_data_in_gpu: 
        info = [
            tensor.xpu(device_id, non_blocking=True) 
            for tensor in info
        ]  
    elif device.type in ["privateuseone", "ocl"] and not cache_data_in_gpu: 
        info = [
            tensor.to(device_id if device.type == "ocl" else device, non_blocking=True) 
            for tensor in info
        ]  
    else: 
        # Fallback tracking for CPU or generalized transfers
        info = [
            tensor.to(device) 
            for tensor in info
        ]
    
    return info

def get_reference(train_loader, reference_path, use_custom_reference = False, pitch_guidance = True, rank = 0, device = "cpu"):
    """
    Fetches reference feature states used to evaluate validation or visual steps. 
    Loads from a static feature directory if specified, otherwise grabs the next standard batch.

    Args:
        train_loader (torch.utils.data.DataLoader): Active fallback training loader pipeline.
        reference_path (str): File folder tracking custom reference features (`feats.npy`).
        use_custom_reference (bool, optional): Force load static file metrics. Defaults to False.
        pitch_guidance (bool, optional): Model relies on explicit F0 values. Defaults to True.
        rank (int, optional): Distributed processing ID. Defaults to 0.
        device (str, optional): Target destination execution string. Defaults to "cpu".

    Returns:
        tuple: Tensor references containing (phone, phone_lengths, pitch, pitchf, sid/wave)
    """

    # Load custom reference numpy files from disk if requested and files exist
    if use_custom_reference and os.path.isfile(os.path.join(reference_path, "feats.npy")):
        # Clean up tracking names using regex
        if rank == 0: logger.info(translations["using_reference"].format(reference_name=re.sub(r'_v\d+_(?:[A-Za-z0-9_]+?)_(True|False)$', '', os.path.basename(reference_path))))
        phone = np.repeat(np.load(os.path.join(reference_path, "feats.npy")), 2, axis=0)

        reference = (
            torch.FloatTensor(phone).unsqueeze(0).to(device),
            torch.LongTensor([phone.shape[0]]).to(device),
            torch.LongTensor(np.load(os.path.join(reference_path, "pitch_coarse.npy"))[:-1]).unsqueeze(0).to(device) if pitch_guidance else None,
            torch.FloatTensor(np.load(os.path.join(reference_path, "pitch_fine.npy"))[:-1]).unsqueeze(0).to(device) if pitch_guidance else None,
            torch.LongTensor([0]).to(device)
        )
    else:
        # Default fallback: dynamically pluck the next available elements inside the train data loader
        info = next(iter(train_loader))
        reference = (info[0].to(device), info[1].to(device))

        if pitch_guidance: reference += (info[2].to(device), info[3].to(device), info[8].to(device))
        else: reference += (None, None, info[6].to(device))
    
    return reference

def check_speaker_dim(hps, checkpoint_path, save_only_latest, pretrainG):
    """
    Infers the dimensional size configuration of the speaker embedding block.
    Checks the hyperparameter configuration, falling back to reading actual tensor shape dimensions 
    from available checkpoints or pretrained generator targets.

    Args:
        hps (HParams): Active hyperparameters object configuration instance.
        checkpoint_path (str): Target directory tracking saved model versions.
        save_only_latest (bool): Flag checking if tracking targets are strictly isolated to latest file.
        pretrainG (str): Filepath location indicating backing base pretrained model.

    Returns:
        int: Total speaker embedding dimension count.
    """

    spk_dim = hps.model.spk_embed_dim
    # Attempt to fall back to speaker identity parameter counts
    try:
        spk_dim = hps.sid
    except Exception as e:
        logger.debug(e)

    # Inspect the model checkpoint architecture weights directly to extract exact embedding dimensions
    try:
        g_path = os.path.join(checkpoint_path, "G_latest.pth")
        last_g = g_path if save_only_latest and os.path.exists(g_path) else latest_checkpoint_path(checkpoint_path, "G_*.pth")
        chk_path = (last_g if last_g else (pretrainG if pretrainG not in ["", "None"] else None))

        if chk_path:
            ckpt = torch.load(chk_path, map_location="cpu", weights_only=True)
            # Read shape dimensions of global embedding weights matrix
            spk_dim = ckpt["model"]["emb_g.weight"].shape[0]
            del ckpt
    except Exception as e:
        logger.debug(e)

    return spk_dim