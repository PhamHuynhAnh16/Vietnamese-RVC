import os
import sys
import torch
import hashlib
import datetime

from collections import OrderedDict

sys.path.append(os.getcwd())

from main.app.variables import logger, translations, config
from main.inference.training.utils import replace_keys_in_dict

def extract_model(
    ckpt, 
    sr, 
    pitch_guidance, 
    name, 
    model_path, 
    epoch, 
    step, 
    version, 
    hps, 
    model_author, 
    vocoder, 
    speakers_id,
    architecture
):
    """
    Extracts the inference-ready weights from a raw training checkpoint, discards 
    unnecessary training blocks (like the posterior encoder `enc_q`), packages structural 
    hyperparameters/metadata, and saves the standalone weights to a targeted disk path.

    Args:
        ckpt (Dict[str, torch.Tensor]): The raw model state dictionary from training.
        sr (int): Target audio sampling rate configuration.
        pitch_guidance (bool): Whether the model uses fundamental frequency (F0) tracking.
        name (str): The logical name assigned to the model.
        model_path (str): File destination path to save the compiled model weight file.
        epoch (int): Total count of epochs processed up to this point.
        step (int): Total global optimization steps performed.
        version (str): Model architecture variant or framework version tag.
        hps (object): Hyperparameters namespace object holding configuration values.
        model_author (str): The name or tag of the model creator/trainer.
        vocoder (str): Type of neural vocoder used for synthesis (e.g., HiFi-GAN).
        speakers_id (int): Dimensional context identifier linked to speaker indexing.
        architecture (str): Target model core neural network style or configuration flavor.
    """

    try:
        logger.info(translations["savemodel"].format(model_dir=model_path, epoch=epoch, step=step))
        # Ensure the destination parent folders exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Initialize ordered tracking; strip out 'enc_q' since the posterior encoder is only needed for training
        opt = OrderedDict(
            weight={
                key: (
                    value if not config.device.startswith("privateuseone") else value.detach().cpu()
                ).to(torch.float16 if config.is_half else torch.float32) 
                for key, value in ckpt.items() 
                if "enc_q" not in key
            }
        )

        # Pack internal structural network dimensions needed for instant architecture reconstruction
        opt["config"] = [
            hps.data.filter_length // 2 + 1, 
            hps.train.segment_size // hps.data.hop_length, 
            hps.model.inter_channels, 
            hps.model.hidden_channels, 
            hps.model.filter_channels, 
            hps.model.n_heads, 
            hps.model.n_layers, 
            hps.model.kernel_size, 
            hps.model.p_dropout, 
            hps.model.resblock, 
            hps.model.resblock_kernel_sizes, 
            hps.model.resblock_dilation_sizes, 
            hps.model.upsample_rates, 
            hps.model.upsample_initial_channel, 
            hps.model.upsample_kernel_sizes, 
            hps.model.spk_embed_dim, 
            hps.model.gin_channels, 
            hps.data.sample_rate
        ]

        # Inject standard model parameters and historical run tracking tags
        opt["epoch"] = f"{epoch}epoch"
        opt["step"] = step
        opt["sr"] = sr
        opt["f0"] = int(pitch_guidance)
        opt["version"] = version
        opt["creation_date"] = datetime.datetime.now().isoformat()
        # Create a unique SHA256 integrity hash footprint string bound to this extraction step
        opt["model_hash"] = hashlib.sha256(f"{str(ckpt)} {epoch} {step} {datetime.datetime.now().isoformat()}".encode()).hexdigest()
        # Append identifying administrative metadata parameters
        opt["model_name"] = name
        opt["author"] = model_author
        opt["vocoder"] = vocoder
        opt["speakers_id"] = speakers_id
        opt["architecture"] = architecture

        # Convert parametrization weight paths back to legacy formats for backward compatibility and save
        torch.save(
            replace_keys_in_dict(
                replace_keys_in_dict(
                    opt, 
                    ".parametrizations.weight.original1", 
                    ".weight_v"
                ), 
                ".parametrizations.weight.original0", ".weight_g"
            ), 
            model_path
        )
    except Exception as e:
        # Catch and report formatting or write exceptions through the centralized logging module
        logger.error(f"{translations['extract_model_error']}: {e}")