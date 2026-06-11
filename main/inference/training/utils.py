import os
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
    import copy

    opt_state = copy.deepcopy(optimizer.state_dict())

    for state in opt_state["state"].values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.detach().cpu()

    return opt_state

def replace_keys_in_dict(d, old_key_part, new_key_part):
    updated_dict = OrderedDict() if isinstance(d, OrderedDict) else {}

    for key, value in d.items():
        updated_dict[(
            key.replace(old_key_part, new_key_part) if isinstance(key, str) else key
        )] = (
            replace_keys_in_dict(value, old_key_part, new_key_part) if isinstance(value, dict) else value
        )
    
    return updated_dict

def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path), translations["not_found_checkpoint"].format(checkpoint_path=checkpoint_path)

    checkpoint_dict = replace_keys_in_dict(
        replace_keys_in_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True), 
            ".weight_v", 
            ".parametrizations.weight.original1"
        ), 
        ".weight_g", 
        ".parametrizations.weight.original0"
    )

    new_state_dict = {
        k: checkpoint_dict["model"].get(k, v) 
        for k, v in (model.module.state_dict() if hasattr(model, "module") else model.state_dict()).items()
    }

    model.module.load_state_dict(new_state_dict, strict=False) if hasattr(model, "module") else model.load_state_dict(new_state_dict, strict=False)
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
    state_dict = (model.module.state_dict() if hasattr(model, "module") else model.state_dict())

    if config.device.startswith("privateuseone"):
        model_state = {k: v.detach().cpu() for k, v in state_dict.items()}
        model_optimizer = optimizer_state_dict_cpu(optimizer)
    else:
        model_state = state_dict
        model_optimizer = optimizer.state_dict()

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
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)

    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)

    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")

    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sample_rate)

def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    checkpoints = sorted(
        glob.glob(
            os.path.join(dir_path, regex)
        ), 
        key=lambda f: int("".join(filter(str.isdigit, f)))
    )
    return checkpoints[-1] if checkpoints else None

def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG

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
    data, sample_rate = sf.read(full_path, dtype=np.float32)
    return torch.FloatTensor(data.astype(np.float32)), sample_rate

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        return [line.strip().split(split) for line in f]
    
class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
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

def get_optimizer(optimizer_choice):
    if optimizer_choice == "AnyPrecisionAdamW" and config.brain:
        from main.library.optimizers.anyprecision_optimizer import AnyPrecisionAdamW

        optimizer_optim = AnyPrecisionAdamW
        scheduler_optim = None
    elif optimizer_choice == "RAdam":
        from torch.optim import RAdam

        optimizer_optim = RAdam
        scheduler_optim = None
    elif optimizer_choice == "AdaBelief":
        from main.library.optimizers.adabelief import AdaBelief

        optimizer_optim = AdaBelief
        scheduler_optim = None
    elif optimizer_choice == "AdaBeliefV2":
        from main.library.optimizers.adabeliefv2 import AdaBeliefV2, get_inverse_sqrt_scheduler

        optimizer_optim = AdaBeliefV2
        scheduler_optim = get_inverse_sqrt_scheduler
    else:
        from torch.optim import AdamW

        optimizer_optim = AdamW
        scheduler_optim = None
    
    return optimizer_optim, scheduler_optim

def cleanup_training(experiment_dir):
    for root, dirs, files in os.walk(experiment_dir, topdown=False):
        for name in files:
            if name.endswith((".0", ".pth", ".index")): os.remove(os.path.join(root, name))

        for name in dirs:
            if name == "eval":
                folder_path = os.path.join(root, name)

                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    if os.path.isfile(item_path): os.remove(item_path)

                os.rmdir(folder_path)

def get_training_data(info, pitch_guidance, energy_use):
    phone, phone_lengths = info[0], info[1]

    if pitch_guidance:
        pitch, pitchf = info[2], info[3]
        spec, spec_lengths, wave, sid = info[4], info[5], info[6], info[8]
        energy = info[9] if energy_use else None
    else:
        pitch = pitchf = None
        spec, spec_lengths, wave, sid = info[2], info[3], info[4], info[6]
        energy = info[7] if energy_use else None
    
    return phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, sid, energy

def transform_tensor_into_cache(device, device_id, cache_data_in_gpu, cache, train_loader):
    if device.type == "cuda" and cache_data_in_gpu:
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
    elif device.type == "xpu" and cache_data_in_gpu:
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
    elif device.type in ["privateuseone", "ocl"] and cache_data_in_gpu:
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
    else: 
        data_iterator = enumerate(train_loader)
    
    return data_iterator, cache

def transforming_computing_devices(info, device, device_id, cache_data_in_gpu):
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
        info = [
            tensor.to(device) 
            for tensor in info
        ]
    
    return info