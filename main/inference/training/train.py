import os
import sys
import glob
import json
import torch
import logging
import argparse
import datetime
import warnings

import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from random import randint
from collections import deque
from contextlib import nullcontext
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from time import time as ttime
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.getcwd())

# Optimize multi-threading backends based on operating system
# Use standard system threads (libuv) on POSIX/Linux, fall back to native threads on Windows
os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"

from main.library.backends import opencl, xpu
from main.app.variables import logger, translations
from main.library.utils import clear_gpu_cache, strtobool
from main.inference.training.extract_model import extract_model
from main.inference.training.data_utils import get_training_dataloader
from main.inference.training.mel_processing import MultiScaleMelSpectrogramLoss, mel_spectrogram_torch, spec_to_mel_torch

from main.library.algorithm import commons
from main.inference.training import utils, losses, detector

from main.app.variables import config as main_config
from main.app.variables import configs as main_configs

if not main_config.debug_mode:
    warnings.filterwarnings("ignore")
    logging.getLogger("torch").setLevel(logging.ERROR)

def parse_arguments():
    """
    Parses CLI configuration parameters for the training workspace, hardware routing, 
    loss function adjustments, and model architecture definitions.
    
    Returns:
        argparse.Namespace: Validated command-line argument object.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--rvc_version", type=str, default="v2")
    parser.add_argument("--save_every_epoch", type=int, required=True)
    parser.add_argument("--save_only_latest", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--save_every_weights", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--total_epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--pitch_guidance", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--g_pretrained_path", type=str, default="")
    parser.add_argument("--d_pretrained_path", type=str, default="")
    parser.add_argument("--overtraining_detector", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--overtraining_threshold", type=int, default=50)
    parser.add_argument("--cleanup", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--cache_data_in_gpu", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--model_author", type=str)
    parser.add_argument("--vocoder", type=str, default="Default")
    parser.add_argument("--checkpointing", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--deterministic", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--benchmark", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--use_custom_reference", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--reference_path", type=str, default="")
    parser.add_argument("--multiscale_mel_loss", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--use_cosine_annealing_lr", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--architecture", type=str, default="RVC")
    parser.add_argument("--filelist_path", type=str, default="")
    parser.add_argument("--config_save_path", type=str, default="")
    parser.add_argument("--spec_dir", type=str, default="")
    parser.add_argument("--eval_dir", type=str, default="")
    parser.add_argument("--cache_spectrogram", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--save_the_pid", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--custom_training", type=lambda x: bool(strtobool(x)), default=False)

    return parser.parse_args()

# Global argument instantiation and tuple unpacking
args = parse_arguments()

(
    model_name, 
    save_every_epoch, 
    total_epoch, 
    pretrainG, 
    pretrainD, 
    version, 
    gpus, 
    batch_size, 
    pitch_guidance, 
    save_only_latest, 
    save_every_weights, 
    cache_data_in_gpu, 
    overtraining_detector, 
    overtraining_threshold, 
    cleanup, 
    model_author, 
    vocoder, 
    checkpointing, 
    optimizer_choice, 
    use_custom_reference, 
    reference_path, 
    multiscale_mel_loss,
    use_cosine_annealing_lr,
    architecture,
    filelist_path,
    config_save_path,
    spec_dir,
    eval_dir,
    cache_spectrogram,
    save_the_pid,
    custom_training
) = (
    args.model_name, 
    args.save_every_epoch, 
    args.total_epoch, 
    args.g_pretrained_path, 
    args.d_pretrained_path, 
    args.rvc_version, 
    args.gpu, 
    args.batch_size, 
    args.pitch_guidance, 
    args.save_only_latest, 
    args.save_every_weights, 
    args.cache_data_in_gpu, 
    args.overtraining_detector, 
    args.overtraining_threshold, 
    args.cleanup, 
    args.model_author, 
    args.vocoder, 
    args.checkpointing, 
    args.optimizer, 
    args.use_custom_reference, 
    args.reference_path, 
    args.multiscale_mel_loss,
    args.use_cosine_annealing_lr,
    args.architecture,
    args.filelist_path,
    args.config_save_path,
    args.spec_dir,
    args.eval_dir,
    args.cache_spectrogram,
    args.save_the_pid,
    args.custom_training
)

# Determine global mixed precision strategies
is_half = main_config.is_half
if main_config.brain: is_half = True # Enforce mixed precision flags automatically if BF16 acceleration is available

# Setup appropriate discriminator target versions based on the chosen architecture and synthesis tools
disc_version = version if vocoder not in ["RefineGAN", "BigVGAN"] else "v3"

if architecture == "SVC":
    disc_version = version if vocoder != "Default" else "v0"
    pitch_guidance = True # SVC processes always depend on explicit source audio pitch tracking patterns

weights_path = main_configs["weights_path"]
logs_path = main_configs["logs_path"]
custom_save_checkpoint_path = None

# Configure file storage contexts. If target identity points to an isolated model name, construct logs path.
if not os.path.exists(model_name): experiment_dir = os.path.join(logs_path, model_name)
else:
    # Explicit absolute workspace paths detected
    experiment_dir = model_name
    model_name = os.path.basename(model_name)
    custom_save_checkpoint_path = weights_path

# Bind execution file structures based on routing priorities
training_file_path = os.path.join(experiment_dir, "training_data.json")
checkpoint_path = experiment_dir if custom_save_checkpoint_path is None else custom_save_checkpoint_path
config_save_path = config_save_path if custom_training else os.path.join(experiment_dir, "config.json")
filelist_path = filelist_path if custom_training else os.path.join(experiment_dir, "filelist.txt")
eval_dir = eval_dir if custom_training else os.path.join(experiment_dir, "eval")
spec_dir = spec_dir if custom_training else None

# Gradient balancing and norm calculation constants
d_lr_coeff = 1.0
g_lr_coeff = 1.0
d_step_per_g_step = 1
use_clip_grad_value = False
grad_norm_optim = commons.clip_grad_value if use_clip_grad_value else commons.grad_norm

# Enable hardware acceleration optimizations depending on backend availability
backend_supported = not main_config.device.startswith(("ocl", "privateuseone")) and not main_config.is_zluda
torch.backends.cudnn.deterministic = args.deterministic if backend_supported else False
torch.backends.cudnn.benchmark = args.benchmark if backend_supported else False
torch.backends.cuda.matmul.allow_tf32 = main_config.tf32 if backend_supported else False
torch.backends.cudnn.allow_tf32 = main_config.tf32 if backend_supported else False

# Tracker initialization parameters
global_step = 0
lowest_value = {"step": 0, "value": float("inf"), "epoch": 0}
loss_gen_history, smoothed_loss_gen_history, loss_disc_history, smoothed_loss_disc_history = [], [], [], []

# Use historical deques to safely track rolling metric configurations
avg_losses = {
    "grad_d_50": deque(maxlen=50), 
    "grad_g_50": deque(maxlen=50), 
    "disc_loss_50": deque(maxlen=50), 
    "adv_loss_50": deque(maxlen=50), 
    "fm_loss_50": deque(maxlen=50), 
    "kl_loss_50": deque(maxlen=50), 
    "mel_loss_50": deque(maxlen=50), 
    "gen_loss_50": deque(maxlen=50)
}

# Load hyperparameters from file configuration profiles
with open(config_save_path, "r", encoding="utf-8") as f:
    config = json.load(f)

config = utils.HParams(**config)
config.data.training_files = filelist_path

def main():
    """
    Orchestrates infrastructure verification, cross-process messaging networks (DDP),
    dataset sample rate validation, and launches parallel sub-process pools.
    """

    global smoothed_loss_gen_history, loss_gen_history, loss_disc_history, smoothed_loss_disc_history, gpus

    # Formulate log dictionary mapped against localized string definitions
    log_data = {
        translations["modelname"]: model_name, 
        translations["save_every_epoch"]: save_every_epoch, 
        translations["total_e"]: total_epoch, 
        translations["dorg"].format(pretrainG=pretrainG, pretrainD=pretrainD): "", 
        translations["training_version"]: version, 
        "Gpu": gpus, 
        translations["batch_size"]: batch_size, 
        translations["training_f0"]: pitch_guidance, 
        translations["save_only_latest"]: save_only_latest, 
        translations["save_every_weights"]: save_every_weights, 
        translations["cache_in_gpu"]: cache_data_in_gpu, 
        translations["overtraining_detector"]: overtraining_detector, 
        translations["threshold"]: overtraining_threshold, 
        translations["cleanup_training"]: cleanup, 
        translations["memory_efficient_training"]: checkpointing, 
        translations["optimizer"]: optimizer_choice, 
        translations["multiscale_mel_loss"]: multiscale_mel_loss,
        translations["model_author"].format(model_author=model_author): "",
        translations["vocoder"]: vocoder,
        translations["cosine_annealing_lr"]: use_cosine_annealing_lr,
        translations["architecture"]: architecture
    }

    for key, value in log_data.items():
        logger.debug(f"{key}: {value}" if value != "" else f"{key} {value}")

    try:
        # Establish localized networking values for multi-GPU process communication
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(randint(20000, 55555))

        # Perform critical verification on dataset sample rate mappings
        wavs = glob.glob(os.path.join(os.path.join(experiment_dir, "sliced_audios"), "*.wav"))
        if wavs:
            _, sr = utils.load_wav_to_torch(wavs[0])
            if sr != config.data.sample_rate:
                logger.warning(translations["training_sr"].format(sr_1=config.data.sample_rate, sr_2=sr))
                sys.exit(1)
        else:
            logger.warning(translations["not_found_dataset"])
            sys.exit(1)

        # Map active runtime hardware device profiles
        device, gpus, n_gpus = utils.get_device(gpus)
        logger.info(translations["use_precision"].format(fp=("BF16" if main_config.brain else "FP16") if is_half else "FP32"))

        # Clear historical states if requested
        if cleanup: utils.cleanup_training(experiment_dir)
        # Restore rolling historical states for early stopping evaluation
        if overtraining_detector and os.path.exists(training_file_path): smoothed_loss_gen_history, loss_gen_history, loss_disc_history, smoothed_loss_disc_history = detector.continue_overtrain_detector(training_file_path)

        def start():
            """Spawns parallel training processes across designated target hardware devices."""

            children = []
            pid_data = {"process_pids": []}
            # Retrieve existing structural configurations if tracking execution PIDs
            if save_the_pid:
                with open(config_save_path, "r", encoding="utf-8") as f:
                    try:
                        pid_data.update(json.load(f))
                    except json.JSONDecodeError:
                        pass

            # Fork individual sub-processes allocated across independent devices
            for rank, device_id in enumerate(gpus):
                subproc = mp.Process(target=run, args=(rank, n_gpus, pretrainG, pretrainD, pitch_guidance, total_epoch, save_every_weights, config, device, device_id, vocoder, checkpointing))
                children.append(subproc)
                subproc.start()
                pid_data["process_pids"].append(subproc.pid)

            # Update the configuration file with the active process IDs
            if save_the_pid: 
                with open(config_save_path, "w", encoding="utf-8") as f:
                    json.dump(pid_data, f, indent=4)

            # Synchronize threads by waiting for all child processes to complete
            for i in range(n_gpus):
                children[i].join()

        start()
    except Exception as e:
        logger.error(f"{translations['training_error']} {e}")
        import traceback
        logger.debug(traceback.format_exc())

class EpochRecorder:
    """Calculates and formats execution performance speeds and durations across epoch bounds."""

    def __init__(self):
        self.last_time = ttime()

    def record(self):
        """
        Calculates time elapsed since the last checkpoint and returns a formatted timestamp.
        
        Returns:
            str: Log-scannable runtime breakdown statement.
        """

        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time

        return translations["time_or_speed_training"].format(current_time=datetime.datetime.now().strftime("%H:%M:%S"), elapsed_time_str=str(datetime.timedelta(seconds=int(round(elapsed_time, 1)))))

def run(rank, n_gpus, pretrainG, pretrainD, pitch_guidance, custom_total_epoch, custom_save_every_weights, config, device, device_id, vocoder, checkpointing):
    """
    Initializes distributed communication backends, instantiates neural model modules,
    restores checkpoints, and manages the training loop across epochs.
    """

    global global_step, smoothed_value_gen, smoothed_value_disc

    smoothed_value_gen, smoothed_value_disc = 0, 0
    # Initialize PyTorch Distributed Process Groups
    dist.init_process_group(
        backend="gloo" if sys.platform == "win32" or device.type not in ["cuda", "xpu"] else ("xccl" if device.type == "xpu" else "nccl"), 
        init_method="env://", 
        world_size=n_gpus if device.type in ["cuda", "xpu"] else 1, 
        rank=rank if device.type in ["cuda", "xpu"] else 0
    )

    # Enforce global pseudo-random seed criteria across active hardware targets
    torch.manual_seed(config.train.seed)
    if device.type == "cuda": torch.cuda.manual_seed(config.train.seed)
    elif device.type == "xpu": torch.xpu.manual_seed(config.train.seed)
    elif device.type == "ocl": opencl.pytorch_ocl.manual_seed_all(config.train.seed)

    # Bind active device visibility scopes per thread
    if device.type == "cuda": torch.cuda.set_device(device_id)
    elif device.type == "xpu": torch.xpu.set_device(device_id)

    # Establish localized logging tools exclusively on the primary rank
    writer_eval = SummaryWriter(log_dir=eval_dir) if rank == 0 else None
    # Instantiate the data pipeline loader
    train_loader = get_training_dataloader(config, spec_dir, cache_spectrogram, pitch_guidance, architecture, batch_size, n_gpus, rank)

    if len(train_loader) < 3:
        logger.warning(translations["not_enough_data"])
        sys.exit(1)

    # Validate speaker dimension metrics
    spk_dim = utils.check_speaker_dim(config, checkpoint_path, save_only_latest, pretrainG)
    config.model.spk_embed_dim = spk_dim

    # Import target structural model blueprints dynamically
    from main.library.algorithm.discriminators import MultiPeriodDiscriminator
    from main.library.algorithm.synthesizers import Synthesizer, SynthesizerSVC

    # Select and initialize model architectures
    net_g, net_d = (
        (
            Synthesizer(
                config.data.filter_length // 2 + 1, 
                config.train.segment_size // config.data.hop_length, 
                **config.model, 
                use_f0=pitch_guidance, 
                sr=config.data.sample_rate, 
                vocoder=vocoder, 
                checkpointing=checkpointing
            )
        ) if architecture == "RVC" else (
            SynthesizerSVC(
                config.data.filter_length // 2 + 1, 
                config.train.segment_size // config.data.hop_length, 
                **config.model, 
                sr=config.data.sample_rate, 
                vocoder=vocoder, 
                checkpointing=checkpointing
            )
        ), 
        MultiPeriodDiscriminator(
            version=disc_version, 
            use_spectral_norm=config.model.use_spectral_norm, 
            checkpointing=checkpointing
        )
    )

    # Migrate models to target hardware acceleration spaces
    net_g, net_d = (net_g.cuda(device_id), net_d.cuda(device_id)) if device.type == "cuda" else (net_g.xpu(device_id), net_d.xpu(device_id)) if device.type == "xpu" else (net_g.to(device), net_d.to(device))
    # Instantiate optimization models
    optim_g, optim_d = utils.get_optimizer(net_g, net_d, optimizer_choice, learning_rate=config.train.learning_rate, betas=config.train.betas, eps=config.train.eps, g_lr_coeff=g_lr_coeff, d_lr_coeff=d_lr_coeff)

    # Bind loss configurations based on configuration criteria
    fn_mel_loss = MultiScaleMelSpectrogramLoss(sample_rate=config.data.sample_rate) if multiscale_mel_loss else torch.nn.L1Loss()
    # Wrap model modules with Distributed Data Parallel handlers
    if device.type.startswith(("cuda", "cpu")): net_g, net_d = (DDP(net_g, device_ids=[device_id]), DDP(net_d, device_ids=[device_id])) if device.type.startswith("cuda") else (DDP(net_g), DDP(net_d))

    scaler_dict = {}
    try:
        if rank == 0: logger.info(translations["start_training"])

        # Locate target checkpoint state paths
        d_path = os.path.join(checkpoint_path, "D_latest.pth") if save_only_latest else utils.latest_checkpoint_path(checkpoint_path, "D_*.pth")
        g_path = os.path.join(checkpoint_path, "G_latest.pth") if save_only_latest else utils.latest_checkpoint_path(checkpoint_path, "G_*.pth")

        # Attempt to load target parameters
        _, _, _, epoch_str, scaler_dict = utils.load_checkpoint(d_path, net_d, optim_d)
        _, _, _, epoch_str, scaler_dict = utils.load_checkpoint(g_path, net_g, optim_g)
        
        if rank == 0: logger.info(translations["load_checkpoint"].format(d_path=d_path, g_path=g_path))
        
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        # Fall back to base foundation files if active checkpoint instances are not found
        check = ["", "None"]
        epoch_str, global_step = 1, 0
        strict = main_configs.get("pretrain_strict", True)

        try:
            if pretrainG not in check:
                if rank == 0: logger.info(translations["import_pretrain"].format(dg="G", pretrain=pretrainG))

                ckptG = torch.load(pretrainG, map_location="cpu", weights_only=True)["model"]
                # Handle SVC speaker embed logic
                if architecture == "SVC" and "emb_g.weight" not in ckptG: ckptG["emb_g.weight"] = net_g.module.emb_g.weight if hasattr(net_g, "module") else net_g.emb_g.weight
                net_g.module.load_state_dict(ckptG, strict=strict) if hasattr(net_g, "module") else net_g.load_state_dict(ckptG, strict=strict)
                del ckptG

            if pretrainD not in check:
                if rank == 0: logger.info(translations["import_pretrain"].format(dg="D", pretrain=pretrainD))

                ckptD = torch.load(pretrainD, map_location="cpu", weights_only=True)["model"]
                net_d.module.load_state_dict(ckptD, strict=strict) if hasattr(net_d, "module") else net_d.load_state_dict(ckptD, strict=strict)
                del ckptD
        except Exception as e:
            logger.error(translations["checkpointing_err"])
            logger.debug(e)
            sys.exit(1)

    # Initialize learning rate decay schedulers
    scheduler_g, scheduler_d = utils.get_scheduler(optim_g, optim_d, optimizer_choice, total_epoch=total_epoch, epoch_str=epoch_str, use_cosine_annealing_lr=use_cosine_annealing_lr, lr_decay=config.train.lr_decay)

    # Initialize hardware gradient scalers for half-precision modes
    if device.type == "xpu" and is_half: xpu.setup_gradscaler()
    scaler = GradScaler(device=device, enabled=is_half and device.type in ["cuda", "xpu"])
    cache = []

    # Restore scaling states if continuing from a saved state
    if len(scaler_dict) > 0: scaler.load_state_dict(scaler_dict)
    # Generate anchor audios for validation tracing
    reference = utils.get_reference(train_loader, reference_path, use_custom_reference=use_custom_reference, pitch_guidance=pitch_guidance, rank=rank, device=device)

    # Execute main epoch processing blocks
    for epoch in range(epoch_str, total_epoch + 1):
        train_and_evaluate(rank, epoch, config, net_g, net_d, optim_g, optim_d, scaler, train_loader, writer_eval, cache, custom_save_every_weights, custom_total_epoch, device, device_id, reference, fn_mel_loss)
        scheduler_g.step(); scheduler_d.step()

def train_and_evaluate(rank, epoch, hps, net_g, net_d, optim_g, optim_d, scaler, train_loader, writer, cache, custom_save_every_weights, custom_total_epoch, device, device_id, reference, fn_mel_loss):
    """Executes step optimizations for both Generator and Discriminator components within an epoch."""

    global global_step, lowest_value, loss_disc, consecutive_increases_gen, consecutive_increases_disc, smoothed_value_gen, smoothed_value_disc

    if epoch == 1:
        lowest_value = {"step": 0, "value": float("inf"), "epoch": 0}
        consecutive_increases_gen, consecutive_increases_disc = 0, 0

    # Align dataloader shuffling tracking with epoch counts for standard setups
    if architecture != "SVC": train_loader.batch_sampler.set_epoch(epoch)
    net_g.train(); net_d.train()

    # Streamline tensor data pipeline allocations
    data_iterator, cache = utils.transform_tensor_into_cache(device, device_id, cache_data_in_gpu, cache, train_loader)
    epoch_recorder = EpochRecorder()

    # Configure mixed precision execution scopes
    autocast_enabled = is_half and device.type in ["cuda", "xpu"]
    autocast_dtype = torch.float32 if not autocast_enabled else torch.bfloat16 if main_config.brain else torch.float16
    autocasts = autocast(device.type, enabled=autocast_enabled, dtype=autocast_dtype) if not device.type.startswith(("ocl", "privateuseone")) else nullcontext()
    
    with tqdm(total=len(train_loader), leave=False) as pbar:
        for _, info in data_iterator:
            info = utils.transforming_computing_devices(info, device, device_id, cache_data_in_gpu)
            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, sid = utils.get_training_data(info, pitch_guidance)

            # GENERATOR FORWARD PASS
            with autocasts:
                y_hat, ids_slice, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                wave = commons.slice_segments(wave, ids_slice * config.data.hop_length, config.train.segment_size, dim=3)

            # DISCRIMINATOR OPTIMIZATION STEP
            for _ in range(d_step_per_g_step):
                with autocasts:
                    y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())

                loss_disc, losses_disc_r, losses_disc_g = losses.discriminator_loss(y_d_hat_r, y_d_hat_g)
                optim_d.zero_grad()

                if autocast_enabled:
                    scaler.scale(loss_disc).backward()
                    scaler.unscale_(optim_d)
                    grad_norm_d = grad_norm_optim(net_d.parameters())
                    scaler.step(optim_d)
                else:
                    loss_disc.backward()
                    grad_norm_d = grad_norm_optim(net_d.parameters())
                    optim_d.step()

            # GENERATOR BACKWARD & LOSSES OPTIMIZATION STEP
            with autocasts:
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)

            y_hat_mel = mel_spectrogram_torch(y_hat.float().squeeze(1), config)

            # Route calculation path using targeted error criteria definitions
            if multiscale_mel_loss: loss_mel = fn_mel_loss(wave, y_hat) * config.train.c_mel / 3.0
            else: loss_mel = fn_mel_loss(mel_spectrogram_torch(wave.float().squeeze(1), config), y_hat_mel) * config.train.c_mel

            # Aggregate total components for multi-task loss calculation
            loss_kl = losses.kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl
            loss_fm = losses.feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = losses.generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

            # Update historical global low bounds
            if loss_gen_all < lowest_value["value"]:  lowest_value = {"step": global_step, "value": loss_gen_all, "epoch": epoch}

            optim_g.zero_grad()
            if autocast_enabled:
                scaler.scale(loss_gen_all).backward()
                scaler.unscale_(optim_g)
                grad_norm_g = grad_norm_optim(net_g.parameters())
                scaler.step(optim_g)
                scaler.update()
            else:
                loss_gen_all.backward()
                grad_norm_g = grad_norm_optim(net_g.parameters())
                optim_g.step()

            global_step += 1
            # Append calculated step parameters to history tracking arrays
            avg_losses["grad_d_50"].append(grad_norm_d)
            avg_losses["grad_g_50"].append(grad_norm_g)
            avg_losses["disc_loss_50"].append(loss_disc.detach())
            avg_losses["adv_loss_50"].append(loss_gen.detach())
            avg_losses["fm_loss_50"].append(loss_fm.detach())
            avg_losses["kl_loss_50"].append(loss_kl.detach())
            avg_losses["mel_loss_50"].append(loss_mel.detach())
            avg_losses["gen_loss_50"].append(loss_gen_all.detach())

            # Export rolling status charts onto summary board records at standard logging boundaries
            if rank == 0 and global_step % 50 == 0:
                scalar_dict = {
                    "grad_avg_50/norm_d": sum(avg_losses["grad_d_50"]) / len(avg_losses["grad_d_50"]),
                    "grad_avg_50/norm_g": sum(avg_losses["grad_g_50"]) / len(avg_losses["grad_g_50"]),
                    "loss_avg_50/d/adv": torch.stack(list(avg_losses["disc_loss_50"])).mean(),
                    "loss_avg_50/g/adv": torch.stack(list(avg_losses["adv_loss_50"])).mean(),
                    "loss_avg_50/g/fm": torch.stack(list(avg_losses["fm_loss_50"])).mean(),
                    "loss_avg_50/g/kl": torch.stack(list(avg_losses["kl_loss_50"])).mean(),
                    "loss_avg_50/g/mel": torch.stack(list(avg_losses["mel_loss_50"])).mean(),
                    "loss_avg_50/g/total": torch.stack(list(avg_losses["gen_loss_50"])).mean()
                }

                utils.summarize(writer=writer, global_step=global_step, scalars=scalar_dict)

            pbar.update(1)

    # Flush active VRAM memory caches safely
    with torch.no_grad():
        clear_gpu_cache()

    # EVALUATION, LOGGING, & EXTRACTION CHECKS
    if rank == 0:
        mel = spec_to_mel_torch(spec, config)
        y_mel = commons.slice_segments(mel, ids_slice, config.train.segment_size // config.data.hop_length, dim=3)

        scalar_dict = {
            "loss/g/total": loss_gen_all, 
            "loss/d/adv": loss_disc, 
            "learning_rate": optim_g.param_groups[0]["lr"], 
            "grad/norm_d": grad_norm_d, 
            "grad/norm_g": grad_norm_g, 
            "loss/g/adv": loss_gen,
            "loss/g/fm": loss_fm, 
            "loss/g/mel": loss_mel, 
            "loss/g/kl": loss_kl
        }

        # Expand data maps with nested sub-loss arrays dynamically
        scalar_dict.update({f"loss/g/{i}": v for i, v in enumerate(losses_gen)})
        scalar_dict.update({f"loss/d_r/{i}": v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({f"loss/d_g/{i}": v for i, v in enumerate(losses_disc_g)})

        image_dict = {
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy().copy()),
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
        }

        # Handle full model saving and validation sample audio synthesis cycles
        if epoch % save_every_epoch == 0:
            with autocasts:
                with torch.no_grad():
                    o = net_g.module.infer(*reference) if hasattr(net_g, "module") else net_g.infer(*reference)

            utils.summarize(writer=writer, global_step=global_step, images=image_dict, scalars=scalar_dict, audios={f"gen/audio_{global_step:07d}": o[0, :, :]}, audio_sample_rate=config.data.sample_rate)
        else:  utils.summarize(writer=writer, global_step=global_step, images=image_dict, scalars=scalar_dict)
    
    model_add, model_del = [], []
    done = False
    
    if rank == 0:
        # Save structural weights when passing interval checkpoints
        if epoch % save_every_epoch == 0:
            checkpoint_suffix = f"{'latest' if save_only_latest else global_step}.pth"

            utils.save_checkpoint(net_g, optim_g, config.train.learning_rate, epoch, os.path.join(checkpoint_path, "G_" + checkpoint_suffix), scaler)
            utils.save_checkpoint(net_d, optim_d, config.train.learning_rate, epoch, os.path.join(checkpoint_path, "D_" + checkpoint_suffix), scaler)

            if custom_save_every_weights: model_add.append(os.path.join(weights_path, f"{model_name}_{epoch}e_{global_step}s.pth"))

        # Monitor divergence characteristics via Automated Early Stopping Checks
        if overtraining_detector and epoch > 1:
            current_loss_disc, current_loss_gen = float(loss_disc), float(lowest_value["value"])
            loss_disc_history.append(current_loss_disc)
            loss_gen_history.append(current_loss_gen)
            
            (
                smoothed_value_disc, 
                smoothed_value_gen, 
                is_overtraining_disc, 
                is_overtraining_gen, 
                consecutive_increases_disc, 
                consecutive_increases_gen
            ) = detector.overtraining_detector(
                smoothed_loss_disc_history, 
                current_loss_disc, 
                smoothed_loss_gen_history, 
                current_loss_gen, 
                overtraining_threshold,
                consecutive_increases_disc,
                consecutive_increases_gen
            )

            if epoch % save_every_epoch == 0: detector.save_to_json(training_file_path, loss_disc_history, smoothed_loss_disc_history, loss_gen_history, smoothed_loss_gen_history)
            # Check early termination flags based on trend thresholds
            if is_overtraining_gen and consecutive_increases_gen == overtraining_threshold or is_overtraining_disc and consecutive_increases_disc == (overtraining_threshold * 2):
                logger.info(translations["overtraining_find"].format(epoch=epoch, smoothed_value_gen=f"{smoothed_value_gen:.3f}", smoothed_value_disc=f"{smoothed_value_disc:.3f}"))
                done = True
            else:
                logger.info(translations["best_epoch"].format(epoch=epoch, smoothed_value_gen=f"{smoothed_value_gen:.3f}", smoothed_value_disc=f"{smoothed_value_disc:.3f}"))
                # Maintain a clean workspace by purging outdated intermediate weight paths
                for file in glob.glob(os.path.join(weights_path, f"{model_name}_*e_*s_best_epoch.pth")):
                    model_del.append(file)

                model_add.append(os.path.join(weights_path, f"{model_name}_{epoch}e_{global_step}s_best_epoch.pth"))
        
        # Standard workflow completion criteria check
        if epoch >= custom_total_epoch:
            logger.info(translations["success_training"].format(epoch=epoch, global_step=global_step, loss_gen_all=round(loss_gen_all.item(), 3)))
            logger.info(translations["training_info"].format(lowest_value_rounded=round(float(lowest_value["value"]), 3), lowest_value_epoch=lowest_value['epoch'], lowest_value_step=lowest_value['step']))
            model_add.append(os.path.join(weights_path, f"{model_name}_{epoch}e_{global_step}s.pth"))
            done = True
            
        # Clean up stale inference models from disk
        for m in model_del:
            os.remove(m)
        
        # Extract targeted production-ready inference weights (.pth)
        if model_add:
            ckpt = (net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict())
            for m in model_add:
                extract_model(
                    ckpt=ckpt, 
                    sr=config.data.sample_rate, 
                    pitch_guidance=pitch_guidance, 
                    name=model_name, 
                    model_path=m, 
                    epoch=epoch, 
                    step=global_step, 
                    version=version, 
                    hps=hps, 
                    model_author=model_author, 
                    vocoder=vocoder, 
                    speakers_id=config.sid,
                    architecture=architecture
                )

        lowest_value_rounded = round(float(lowest_value["value"]), 3)
        # Output generalized execution summary statistics depending on tracking contexts
        if epoch > 1 and overtraining_detector: 
            logger.info(
                translations["model_training_info"].format(
                    model_name=model_name, 
                    epoch=epoch, 
                    global_step=global_step, 
                    epoch_recorder=epoch_recorder.record(), 
                    lowest_value_rounded=lowest_value_rounded, 
                    lowest_value_epoch=lowest_value['epoch'], 
                    lowest_value_step=lowest_value['step'], 
                    remaining_epochs_gen=overtraining_threshold - consecutive_increases_gen, 
                    remaining_epochs_disc=(overtraining_threshold * 2) - consecutive_increases_disc, 
                    smoothed_value_gen=f"{smoothed_value_gen:.3f}", 
                    smoothed_value_disc=f"{smoothed_value_disc:.3f}"
                )
            )
        elif epoch > 1 and not overtraining_detector: 
            logger.info(
                translations["model_training_info_2"].format(
                    model_name=model_name, 
                    epoch=epoch, 
                    global_step=global_step, 
                    epoch_recorder=epoch_recorder.record(), 
                    lowest_value_rounded=lowest_value_rounded, 
                    lowest_value_epoch=lowest_value['epoch'], 
                    lowest_value_step=lowest_value['step']
                )
            )
        else: logger.info(translations["model_training_info_3"].format(model_name=model_name, epoch=epoch, global_step=global_step, epoch_recorder=epoch_recorder.record()))

        logger.debug(f"loss_gen_all: {loss_gen_all} loss_gen: {loss_gen} loss_fm: {loss_fm} loss_mel: {loss_mel} loss_kl: {loss_kl}")

        # Post-training cleanup routine upon triggering termination flags
        if done: 
            if save_the_pid:
                with open(config_save_path, "r", encoding="utf-8") as pid_file:
                    pid_data = json.load(pid_file)

                with open(config_save_path, "w", encoding="utf-8") as pid_file:
                    pid_data.pop("process_pids", None)
                    json.dump(pid_data, pid_file, indent=4)

                if os.path.exists(os.path.join(experiment_dir, "train_pid.txt")): os.remove(os.path.join(experiment_dir, "train_pid.txt"))

            sys.exit(0)

        with torch.no_grad():
            clear_gpu_cache()

if __name__ == "__main__": 
    # Enforce spawn multiprocessing method for thread-safe cross-GPU allocation bounds
    mp.set_start_method("spawn")
    main()