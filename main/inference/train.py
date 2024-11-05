import os
import re
import sys
import glob
import json
import torch
import logging
import hashlib
import argparse
import datetime
import warnings
import logging.handlers

import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from time import time as ttime
from scipy.io.wavfile import read
from collections import OrderedDict
from random import randint, shuffle
 
from torch.nn import functional as F
from distutils.util import strtobool
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from librosa.filters import mel as librosa_mel_fn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

current_dir = os.getcwd()
sys.path.append(current_dir)

from main.library.algorithm.residuals import LRELU_SLOPE
from main.library.algorithm.synthesizers import Synthesizer
from main.library.algorithm.commons import get_padding, slice_segments, clip_grad_value

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.getLogger("torch").setLevel(logging.ERROR)
MATPLOTLIB_FLAG = False

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

def parse_arguments() -> tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--rvc_version", type=str, default="v2")
    parser.add_argument("--save_every_epoch", type=int, required=True)
    parser.add_argument("--save_only_latest", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--save_every_weights", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--total_epoch", type=int, default=300)
    parser.add_argument("--sample_rate", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--pitch_guidance", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--g_pretrained_path", type=str, default="")
    parser.add_argument("--d_pretrained_path", type=str, default="")
    parser.add_argument("--overtraining_detector", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--overtraining_threshold", type=int, default=50)
    parser.add_argument("--sync_graph", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--cache_data_in_gpu", type=lambda x: bool(strtobool(x)), default=False)

    args = parser.parse_args()
    return args


args = parse_arguments()

model_name = args.model_name
save_every_epoch = args.save_every_epoch
total_epoch = args.total_epoch
pretrainG = args.g_pretrained_path
pretrainD = args.d_pretrained_path
version = args.rvc_version
gpus = args.gpu
batch_size = args.batch_size
sample_rate = args.sample_rate
pitch_guidance = args.pitch_guidance
save_only_latest = args.save_only_latest
save_every_weights = args.save_every_weights
cache_data_in_gpu = args.cache_data_in_gpu
overtraining_detector = args.overtraining_detector
overtraining_threshold = args.overtraining_threshold
sync_graph = args.sync_graph

experiment_dir = os.path.join(current_dir, "assets", "logs", model_name)
config_save_path = os.path.join(experiment_dir, "config.json")

os.environ["CUDA_VISIBLE_DEVICES"] = gpus.replace("-", ",")
n_gpus = len(gpus.split("-"))

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

global_step = 0
last_loss_gen_all = 0
overtrain_save_epoch = 0

loss_gen_history = []
smoothed_loss_gen_history = []
loss_disc_history = []
smoothed_loss_disc_history = []

lowest_value = {"step": 0, "value": float("inf"), "epoch": 0}
training_file_path = os.path.join(experiment_dir, "training_data.json")

with open(config_save_path, "r") as f:
    config = json.load(f)

config = HParams(**config)
config.data.training_files = os.path.join(experiment_dir, "filelist.txt")

log_file = os.path.join(experiment_dir, "train.log")

logger = logging.getLogger(__name__)

if logger.hasHandlers(): logger.handlers.clear()
else:  
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

logger.debug(f"Tên mô hình: {model_name}")
logger.debug(f"Lưu mô hình sau: {save_every_epoch} kỷ nguyên")
logger.debug(f"Tổng số kỷ nguyên huấn luyện: {total_epoch} kỷ nguyên")
logger.debug(f"Pretrain G: {pretrainG} | Pretrain D: {pretrainD}")
logger.debug(f"Phiên bản mô hình: {version}")
logger.debug(f"Gpu: {gpus}")
logger.debug(f"Kích thước lô: {batch_size}")
logger.debug(f"Tốc độ lấy mẫu: {sample_rate}")
logger.debug(f"Huấn luyện cao độ: {pitch_guidance}")
logger.debug(f"Chỉ lưu mô hình cuối: {save_only_latest}")
logger.debug(f"Lưu mọi mô hình: {save_every_weights}")
logger.debug(f"Lưu lên bộ nhớ Gpu: {cache_data_in_gpu}")
logger.debug(f"Kiểm tra huấn luyện quá sức: {overtraining_detector}")
logger.debug(f"Ngưỡng huấn luyện quá sức: {overtraining_threshold}")
logger.debug(f"Đồng bộ hóa biểu đồ: {sync_graph}")

def main():
    global training_file_path, last_loss_gen_all, smoothed_loss_gen_history, loss_gen_history, loss_disc_history, smoothed_loss_disc_history, overtrain_save_epoch
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        n_gpus = 1
    else:
        device = torch.device("cpu")
        n_gpus = 1

    def start():
        children = []

        for i in range(n_gpus):
            subproc = mp.Process(target=run, args=(i, n_gpus, experiment_dir, pretrainG, pretrainD, pitch_guidance, custom_total_epoch, custom_save_every_weights, config, device))
            children.append(subproc)
            subproc.start()

        for i in range(n_gpus):
            children[i].join()

    def load_from_json(file_path):
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)

                return (
                    data.get("loss_disc_history", []),
                    data.get("smoothed_loss_disc_history", []),
                    data.get("loss_gen_history", []),
                    data.get("smoothed_loss_gen_history", []),
                )
            
        return [], [], [], []

    def continue_overtrain_detector(training_file_path):
        if overtraining_detector:
            if os.path.exists(training_file_path):
                (
                    loss_disc_history,
                    smoothed_loss_disc_history,
                    loss_gen_history,
                    smoothed_loss_gen_history,
                ) = load_from_json(training_file_path)


    n_gpus = torch.cuda.device_count()

    if not torch.cuda.is_available() and torch.backends.mps.is_available(): n_gpus = 1

    if n_gpus < 1:
        logger.warning("Không phát hiện thấy GPU, hoàn nguyên về CPU (không khuyến nghị)")
        n_gpus = 1

    if sync_graph:
        logger.debug("Biểu đồ đồng bộ hiện đã được kích hoạt! Khi bật biểu đồ đồng bộ, mô hình sẽ trải qua một giai đoạn huấn luyện duy nhất. Sau khi các biểu đồ được đồng bộ hóa, quá trình huấn luyện sẽ tiếp tục với số kỷ nguyên được chỉ định trước đó.")
        custom_total_epoch = 1
        custom_save_every_weights = True

        start()

        model_config_file = os.path.join(experiment_dir, "config.json")
        rvc_config_file = os.path.join(current_dir, "main", "configs", version, str(sample_rate) + ".json")

        if not os.path.exists(rvc_config_file): rvc_config_file = os.path.join(current_dir, "main", "configs", "v1", str(sample_rate) + ".json")

        pattern = rf"{os.path.basename(model_name)}_(\d+)e_(\d+)s\.pth"

        for filename in os.listdir(experiment_dir):
            match = re.match(pattern, filename)

            if match: steps = int(match.group(2))

        def edit_config(config_file):
            with open(config_file, "r", encoding="utf8") as json_file:
                config_data = json.load(json_file)

            config_data["train"]["log_interval"] = steps

            with open(config_file, "w", encoding="utf8") as json_file:
                json.dump(config_data, json_file, indent=2, separators=(",", ": "), ensure_ascii=False)

        edit_config(model_config_file)
        edit_config(rvc_config_file)


        for root, dirs, files in os.walk(os.path.join(current_dir, "assets", "logs", model_name), topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                file_name, file_extension = os.path.splitext(name)

                if file_extension == ".0": os.remove(file_path)
                elif ("D" in name or "G" in name) and file_extension == ".pth": os.remove(file_path)
                elif ("added" in name or "trained" in name) and file_extension == ".index": os.remove(file_path)

            for name in dirs:
                if name == "eval":
                    folder_path = os.path.join(root, name)

                    for item in os.listdir(folder_path):
                        item_path = os.path.join(folder_path, item)
                        if os.path.isfile(item_path): os.remove(item_path)

                    os.rmdir(folder_path)

        logger.info("Đồ thị đã được đồng bộ hóa thành công!")
        custom_total_epoch = total_epoch
        custom_save_every_weights = save_every_weights

        continue_overtrain_detector(training_file_path)
        start()
    else:
        custom_total_epoch = total_epoch
        custom_save_every_weights = save_every_weights

        continue_overtrain_detector(training_file_path)
        start()


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG

    if not MATPLOTLIB_FLAG:
        plt.switch_backend("Agg")

        MATPLOTLIB_FLAG = True

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")

    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sample_rate=22050):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)

    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)

    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")

    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sample_rate)


def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path), f"Không tìm thấy tệp điểm đã lưu: {checkpoint_path}"

    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_dict = replace_keys_in_dict(replace_keys_in_dict(checkpoint_dict, ".weight_v", ".parametrizations.weight.original1"), ".weight_g", ".parametrizations.weight.original0")

    model_state_dict = (model.module.state_dict() if hasattr(model, "module") else model.state_dict())
    new_state_dict = {k: checkpoint_dict["model"].get(k, v) for k, v in model_state_dict.items()}

    if hasattr(model, "module"): model.module.load_state_dict(new_state_dict, strict=False)
    else: model.load_state_dict(new_state_dict, strict=False)

    if optimizer and load_opt == 1: optimizer.load_state_dict(checkpoint_dict.get("optimizer", {}))

    logger.debug(f"Đã tải lại điểm đã lưu '{checkpoint_path}' (kỷ nguyên {checkpoint_dict['iteration']})")

    return (
        model,
        optimizer,
        checkpoint_dict.get("learning_rate", 0),
        checkpoint_dict["iteration"],
    )


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    state_dict = (model.module.state_dict() if hasattr(model, "module") else model.state_dict())

    checkpoint_data = {
        "model": state_dict,
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate,
    }

    torch.save(checkpoint_data, checkpoint_path)

    old_version_path = checkpoint_path.replace(".pth", "_old_version.pth")

    checkpoint_data = replace_keys_in_dict(replace_keys_in_dict(checkpoint_data, ".parametrizations.weight.original1", ".weight_v"), ".parametrizations.weight.original0", ".weight_g")

    torch.save(checkpoint_data, old_version_path)

    os.replace(old_version_path, checkpoint_path)
    logger.info(f"Đã lưu mô hình '{checkpoint_path}' (kỷ nguyên {iteration})")


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    checkpoints = sorted(glob.glob(os.path.join(dir_path, regex)), key=lambda f: int("".join(filter(str.isdigit, f))))

    return checkpoints[-1] if checkpoints else None


def load_wav_to_torch(full_path):
    sample_rate, data = read(full_path)
    
    return torch.FloatTensor(data.astype(np.float32)), sample_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        return [line.strip().split(split) for line in f]


def feature_loss(fmap_r, fmap_g):
    loss = 0

    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []

    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []

    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)

    return l


class TextAudioLoaderMultiNSFsid(torch.utils.data.Dataset):
    def __init__(self, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(hparams.training_files)
        self.max_wav_value = hparams.max_wav_value
        self.sample_rate = hparams.sample_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sample_rate = hparams.sample_rate
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 5000)
        self._filter()

    def _filter(self):
        audiopaths_and_text_new = []
        lengths = []

        for audiopath, text, pitch, pitchf, dv in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, text, pitch, pitchf, dv])
                lengths.append(os.path.getsize(audiopath) // (3 * self.hop_length))

        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_sid(self, sid):
        try:
            sid = torch.LongTensor([int(sid)])
        except ValueError as error:
            logger.error(f"Lỗi chuyển đổi ID người nói '{sid}' thành số nguyên. Ngoại lệ: {error}")
            sid = torch.LongTensor([0])

        return sid

    def get_audio_text_pair(self, audiopath_and_text):
        file = audiopath_and_text[0]
        phone = audiopath_and_text[1]
        pitch = audiopath_and_text[2]
        pitchf = audiopath_and_text[3]
        dv = audiopath_and_text[4]

        phone, pitch, pitchf = self.get_labels(phone, pitch, pitchf)
        spec, wav = self.get_audio(file)
        dv = self.get_sid(dv)

        len_phone = phone.size()[0]
        len_spec = spec.size()[-1]
        if len_phone != len_spec:
            len_min = min(len_phone, len_spec)
            len_wav = len_min * self.hop_length

            spec = spec[:, :len_min]
            wav = wav[:, :len_wav]

            phone = phone[:len_min, :]
            pitch = pitch[:len_min]
            pitchf = pitchf[:len_min]

        return (spec, wav, phone, pitch, pitchf, dv)

    def get_labels(self, phone, pitch, pitchf):
        phone = np.load(phone)
        phone = np.repeat(phone, 2, axis=0)

        pitch = np.load(pitch)
        pitchf = np.load(pitchf)

        n_num = min(phone.shape[0], 900)
        phone = phone[:n_num, :]

        pitch = pitch[:n_num]
        pitchf = pitchf[:n_num]

        phone = torch.FloatTensor(phone)

        pitch = torch.LongTensor(pitch)
        pitchf = torch.FloatTensor(pitchf)

        return phone, pitch, pitchf

    def get_audio(self, filename):
        audio, sample_rate = load_wav_to_torch(filename)

        if sample_rate != self.sample_rate: raise ValueError(f"{sample_rate} Tỉ lệ mẫu không khớp với mục tiêu {self.sample_rate} Tỉ lệ mẫu")

        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")

        if os.path.exists(spec_filename):
            try:
                spec = torch.load(spec_filename)
            except Exception as error:
                logger.error(f"Đã xảy ra lỗi khi nhận thông số kỹ thuật từ {spec_filename}: {error}")
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
                spec = torch.squeeze(spec, 0)

                torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        else:
            spec = spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.hop_length,
                self.win_length,
                center=False,
            )
            spec = torch.squeeze(spec, 0)

            torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        return spec, audio_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollateMultiNSFsid:
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort(torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True)

        max_spec_len = max([x[0].size(1) for x in batch])
        max_wave_len = max([x[1].size(1) for x in batch])

        spec_lengths = torch.LongTensor(len(batch))
        wave_lengths = torch.LongTensor(len(batch))
        spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        wave_padded = torch.FloatTensor(len(batch), 1, max_wave_len)

        spec_padded.zero_()
        wave_padded.zero_()

        max_phone_len = max([x[2].size(0) for x in batch])

        phone_lengths = torch.LongTensor(len(batch))
        phone_padded = torch.FloatTensor(len(batch), max_phone_len, batch[0][2].shape[1])
        pitch_padded = torch.LongTensor(len(batch), max_phone_len)
        pitchf_padded = torch.FloatTensor(len(batch), max_phone_len)

        phone_padded.zero_()
        pitch_padded.zero_()
        pitchf_padded.zero_()
        sid = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            spec = row[0]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wave = row[1]
            wave_padded[i, :, : wave.size(1)] = wave
            wave_lengths[i] = wave.size(1)

            phone = row[2]
            phone_padded[i, : phone.size(0), :] = phone
            phone_lengths[i] = phone.size(0)

            pitch = row[3]
            pitch_padded[i, : pitch.size(0)] = pitch
            pitchf = row[4]
            pitchf_padded[i, : pitchf.size(0)] = pitchf

            sid[i] = row[5]

        return (
            phone_padded,
            phone_lengths,
            pitch_padded,
            pitchf_padded,
            spec_padded,
            spec_lengths,
            wave_padded,
            wave_lengths,
            sid,
        )


class TextAudioLoader(torch.utils.data.Dataset):
    def __init__(self, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(hparams.training_files)
        self.max_wav_value = hparams.max_wav_value
        self.sample_rate = hparams.sample_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sample_rate = hparams.sample_rate
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 5000)
        self._filter()

    def _filter(self):
        audiopaths_and_text_new = []
        lengths = []

        for entry in self.audiopaths_and_text:
            if len(entry) >= 3:
                audiopath, text, dv = entry[:3]

                if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                    audiopaths_and_text_new.append([audiopath, text, dv])
                    lengths.append(os.path.getsize(audiopath) // (3 * self.hop_length))

        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_sid(self, sid):
        try:
            sid = torch.LongTensor([int(sid)])
        except ValueError as error:
            logger.error(f"Lỗi chuyển đổi ID người nói '{sid}' thành số nguyên. Ngoại lệ: {error}")
            sid = torch.LongTensor([0])

        return sid

    def get_audio_text_pair(self, audiopath_and_text):
        file = audiopath_and_text[0]
        phone = audiopath_and_text[1]
        dv = audiopath_and_text[2]

        phone = self.get_labels(phone)
        spec, wav = self.get_audio(file)
        dv = self.get_sid(dv)

        len_phone = phone.size()[0]
        len_spec = spec.size()[-1]

        if len_phone != len_spec:
            len_min = min(len_phone, len_spec)
            len_wav = len_min * self.hop_length
            spec = spec[:, :len_min]
            wav = wav[:, :len_wav]
            phone = phone[:len_min, :]

        return (spec, wav, phone, dv)

    def get_labels(self, phone):
        phone = np.load(phone)
        phone = np.repeat(phone, 2, axis=0)
        n_num = min(phone.shape[0], 900)
        phone = phone[:n_num, :]
        phone = torch.FloatTensor(phone)
        return phone

    def get_audio(self, filename):
        audio, sample_rate = load_wav_to_torch(filename)

        if sample_rate != self.sample_rate: raise ValueError(f"{sample_rate} Tỉ lệ mẫu không phù hợp với mục tiêu {self.sample_rate} Tỉ lệ mẫu")

        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)

        spec_filename = filename.replace(".wav", ".spec.pt")

        if os.path.exists(spec_filename):
            try:
                spec = torch.load(spec_filename)
            except Exception as error:
                logger.error(f"Đã xảy ra lỗi khi nhận thông số kỹ thuật từ {spec_filename}: {error}")
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
                spec = torch.squeeze(spec, 0)

                torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        else:
            spec = spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.hop_length,
                self.win_length,
                center=False,
            )
            spec = torch.squeeze(spec, 0)

            torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        return spec, audio_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate:
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort(torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True)

        max_spec_len = max([x[0].size(1) for x in batch])
        max_wave_len = max([x[1].size(1) for x in batch])

        spec_lengths = torch.LongTensor(len(batch))
        wave_lengths = torch.LongTensor(len(batch))

        spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        wave_padded = torch.FloatTensor(len(batch), 1, max_wave_len)

        spec_padded.zero_()
        wave_padded.zero_()

        max_phone_len = max([x[2].size(0) for x in batch])

        phone_lengths = torch.LongTensor(len(batch))
        phone_padded = torch.FloatTensor(len(batch), max_phone_len, batch[0][2].shape[1])

        phone_padded.zero_()
        sid = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            spec = row[0]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wave = row[1]
            wave_padded[i, :, : wave.size(1)] = wave
            wave_lengths[i] = wave.size(1)

            phone = row[2]
            phone_padded[i, : phone.size(0), :] = phone
            phone_lengths[i] = phone.size(0)

            sid[i] = row[3]

        return (
            phone_padded,
            phone_lengths,
            spec_padded,
            spec_lengths,
            wave_padded,
            wave_lengths,
            sid,
        )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]

        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1: buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, -1, -1):  
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []

        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)

        return buckets, num_samples_per_bucket

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []

        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []

        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            rem = num_samples_bucket - len_bucket

            ids_bucket = (ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[: (rem % len_bucket)])
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size : (j + 1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]

        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None: hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2

            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]: return mid
            elif x <= self.boundaries[mid]: return self._bisect(x, lo, mid)
            else: return self._bisect(x, mid + 1, hi)

        else: return -1

    def __len__(self):
        return self.num_samples // self.batch_size


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11, 17]
        self.discriminators = torch.nn.ModuleList([DiscriminatorS(use_spectral_norm=use_spectral_norm)] + [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MultiPeriodDiscriminatorV2(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminatorV2, self).__init__()
        periods = [2, 3, 5, 7, 11, 17, 23, 37]
        self.discriminators = torch.nn.ModuleList([DiscriminatorS(use_spectral_norm=use_spectral_norm)] + [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = torch.nn.ModuleList([norm_f(torch.nn.Conv1d(1, 16, 15, 1, padding=7)), norm_f(torch.nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)), norm_f(torch.nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)), norm_f(torch.nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)), norm_f(torch.nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)), norm_f(torch.nn.Conv1d(1024, 1024, 5, 1, padding=2))])
        self.conv_post = norm_f(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []

        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        in_channels = [1, 32, 128, 512, 1024]
        out_channels = [32, 128, 512, 1024, 1024]

        self.convs = torch.nn.ModuleList(
            [
                norm_f(
                    torch.nn.Conv2d(
                        in_ch,
                        out_ch,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                )
                for in_ch, out_ch in zip(in_channels, out_channels)
            ]
        )

        self.conv_post = norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []
        b, c, t = x.shape

        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")

        x = x.view(b, c, -1, self.period)

        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time = round(elapsed_time, 1)
        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        return f"thời gian={current_time} | tốc độ huấn luyện={elapsed_time_str}"
    

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes):
    return dynamic_range_decompression_torch(magnitudes)


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, hop_size, win_size, center=False):
    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window: hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect")

    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device], center=center, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)

    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)
    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sample_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)

    melspec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    melspec = spectral_normalize_torch(melspec)
    return melspec


def mel_spectrogram_torch(y, n_fft, num_mels, sample_rate, hop_size, win_size, fmin, fmax, center=False):
    spec = spectrogram_torch(y, n_fft, hop_size, win_size, center)

    melspec = spec_to_mel_torch(spec, n_fft, num_mels, sample_rate, fmin, fmax)

    return melspec


def replace_keys_in_dict(d, old_key_part, new_key_part):
    updated_dict = OrderedDict() if isinstance(d, OrderedDict) else {}

    for key, value in d.items():
        new_key = (key.replace(old_key_part, new_key_part) if isinstance(key, str) else key)
        updated_dict[new_key] = (replace_keys_in_dict(value, old_key_part, new_key_part) if isinstance(value, dict) else value)
        
    return updated_dict


def extract_model(ckpt, sr, pitch_guidance, name, model_dir, epoch, step, version, hps):
    try:
        logger.info(f"Đã lưu mô hình '{model_dir}' (kỷ nguyên {epoch} và bước {step})")

        model_dir_path = os.path.dirname(model_dir)
        os.makedirs(model_dir_path, exist_ok=True)

        if "best_epoch" in model_dir: pth_file = f"{name}_{epoch}e_{step}s_best_epoch.pth"
        else: pth_file = f"{name}_{epoch}e_{step}s.pth"

        pth_file_old_version_path = os.path.join(model_dir_path, f"{pth_file}_old_version.pth")

        opt = OrderedDict(weight={key: value.half() for key, value in ckpt.items() if "enc_q" not in key})

        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            32,
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
            hps.data.sample_rate,
        ]

        opt["epoch"] = f"{epoch}epoch"
        opt["step"] = step
        opt["sr"] = sr
        opt["f0"] = int(pitch_guidance)
        opt["version"] = version
        opt["creation_date"] = datetime.datetime.now().isoformat()

        hash_input = f"{str(ckpt)} {epoch} {step} {datetime.datetime.now().isoformat()}"
        model_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        opt["model_hash"] = model_hash

        torch.save(opt, os.path.join(model_dir_path, pth_file))

        model = torch.load(model_dir, map_location=torch.device("cpu"))

        torch.save(replace_keys_in_dict(replace_keys_in_dict(model, ".parametrizations.weight.original1", ".weight_v"), ".parametrizations.weight.original0", ".weight_g"), pth_file_old_version_path)

        os.remove(model_dir)
        os.rename(pth_file_old_version_path, model_dir)

    except Exception as error:
        logger.error(f"Đã xảy ra lỗi khi trích xuất mô hình: {error}")


def run(rank, n_gpus, experiment_dir, pretrainG, pretrainD, pitch_guidance, custom_total_epoch, custom_save_every_weights, config, device):
    global global_step

    if rank == 0:
        writer = SummaryWriter(log_dir=experiment_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(experiment_dir, "eval"))

    dist.init_process_group(backend="gloo", init_method="env://", world_size=n_gpus, rank=rank)
    torch.manual_seed(config.train.seed)

    if torch.cuda.is_available(): torch.cuda.set_device(rank)

    train_dataset = TextAudioLoaderMultiNSFsid(config.data)

    train_sampler = DistributedBucketSampler(train_dataset, batch_size * n_gpus, [100, 200, 300, 400, 500, 600, 700, 800, 900], num_replicas=n_gpus, rank=rank, shuffle=True)

    collate_fn = TextAudioCollateMultiNSFsid()

    train_loader = DataLoader(train_dataset, num_workers=4, shuffle=False, pin_memory=True, collate_fn=collate_fn, batch_sampler=train_sampler, persistent_workers=True, prefetch_factor=8)

    net_g = Synthesizer(config.data.filter_length // 2 + 1, config.train.segment_size // config.data.hop_length, **config.model, use_f0=pitch_guidance == True, is_half=config.train.fp16_run and device.type == "cuda", sr=sample_rate).to(device)

    if torch.cuda.is_available(): net_g = net_g.cuda(rank)

    if version == "v1": net_d = MultiPeriodDiscriminator(config.model.use_spectral_norm)
    else: net_d = MultiPeriodDiscriminatorV2(config.model.use_spectral_norm)

    if torch.cuda.is_available(): net_d = net_d.cuda(rank)

    optim_g = torch.optim.AdamW(net_g.parameters(), config.train.learning_rate, betas=config.train.betas, eps=config.train.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), config.train.learning_rate, betas=config.train.betas, eps=config.train.eps)

    if torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])
    else:
        net_g = DDP(net_g)
        net_d = DDP(net_d)

    try:
        logger.info("Bắt đầu huấn luyện")

        _, _, _, epoch_str = load_checkpoint(latest_checkpoint_path(experiment_dir, "D_*.pth"), net_d, optim_d)
        _, _, _, epoch_str = load_checkpoint(latest_checkpoint_path(experiment_dir, "G_*.pth"), net_g, optim_g)

        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)

    except:
        epoch_str = 1
        global_step = 0

        if pretrainG != "":
            if rank == 0: logger.info(f"Đã nạp huấn luyện trước (G) '{pretrainG}'")

            if hasattr(net_g, "module"): net_g.module.load_state_dict(torch.load(pretrainG, map_location="cpu")["model"])
            else: net_g.load_state_dict(torch.load(pretrainG, map_location="cpu")["model"])
        else: logger.warning(f"Sẽ không có huấn luyện trước (G) được sử dụng")

        if pretrainD != "":
            if rank == 0: logger.info(f"Đã nạp huấn luyện trước (D) '{pretrainD}'")

            if hasattr(net_d, "module"): net_d.module.load_state_dict(torch.load(pretrainD, map_location="cpu")["model"])
            else: net_d.load_state_dict(torch.load(pretrainD, map_location="cpu")["model"])
        else: logger.warning(f"Sẽ không có huấn luyện trước (D) được sử dụng")

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=config.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=config.train.lr_decay, last_epoch=epoch_str - 2)

    optim_d.step()
    optim_g.step()

    scaler = GradScaler(enabled=config.train.fp16_run)

    cache = []

    for info in train_loader:
        phone, phone_lengths, pitch, pitchf, _, _, _, _, sid = info
        reference = (
            phone.to(device),
            phone_lengths.to(device),
            pitch.to(device) if pitch_guidance else None,
            pitchf.to(device) if pitch_guidance else None,
            sid.to(device),
        )
        break

    for epoch in range(epoch_str, total_epoch + 1):
        if rank == 0: train_and_evaluate(rank, epoch, config, [net_g, net_d], [optim_g, optim_d], scaler, [train_loader, None], [writer, writer_eval], cache, custom_save_every_weights, custom_total_epoch, device, reference)
        else: train_and_evaluate(rank, epoch, config, [net_g, net_d], [optim_g, optim_d], scaler, [train_loader, None], None, cache, custom_save_every_weights, custom_total_epoch, device, reference)

        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, scaler, loaders, writers, cache, custom_save_every_weights, custom_total_epoch, device, reference):
    global global_step, lowest_value, loss_disc, consecutive_increases_gen, consecutive_increases_disc

    if epoch == 1:
        lowest_value = {"step": 0, "value": float("inf"), "epoch": 0}
        last_loss_gen_all = 0.0
        consecutive_increases_gen = 0
        consecutive_increases_disc = 0

    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader = loaders[0] if loaders is not None else None

    if writers is not None: writer = writers[0]

    train_loader.batch_sampler.set_epoch(epoch)

    net_g.train()
    net_d.train()

    if device.type == "cuda" and cache_data_in_gpu:
        data_iterator = cache

        if cache == []:
            for batch_idx, info in enumerate(train_loader):
                (
                    phone,
                    phone_lengths,
                    pitch,
                    pitchf,
                    spec,
                    spec_lengths,
                    wave,
                    wave_lengths,
                    sid,
                ) = info
                cache.append(
                    (batch_idx, (
                        phone.cuda(rank, non_blocking=True),
                        phone_lengths.cuda(rank, non_blocking=True),
                        (pitch.cuda(rank, non_blocking=True) if pitch_guidance else None),
                        (pitchf.cuda(rank, non_blocking=True) if pitch_guidance else None),
                        spec.cuda(rank, non_blocking=True),
                        spec_lengths.cuda(rank, non_blocking=True),
                        wave.cuda(rank, non_blocking=True),
                        wave_lengths.cuda(rank, non_blocking=True),
                        sid.cuda(rank, non_blocking=True),
                        ),
                    ))
        else: shuffle(cache)
    else: data_iterator = enumerate(train_loader)

    epoch_recorder = EpochRecorder()

    with tqdm(total=len(train_loader), leave=False) as pbar:
        for batch_idx, info in data_iterator:
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info
            if device.type == "cuda" and not cache_data_in_gpu:
                phone = phone.cuda(rank, non_blocking=True)
                phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
                pitch = pitch.cuda(rank, non_blocking=True) if pitch_guidance else None
                pitchf = (pitchf.cuda(rank, non_blocking=True) if pitch_guidance else None)
                sid = sid.cuda(rank, non_blocking=True)
                spec = spec.cuda(rank, non_blocking=True)
                spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
                wave = wave.cuda(rank, non_blocking=True)
                wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
            else:
                phone = phone.to(device)
                phone_lengths = phone_lengths.to(device)
                pitch = pitch.to(device) if pitch_guidance else None
                pitchf = pitchf.to(device) if pitch_guidance else None
                sid = sid.to(device)
                spec = spec.to(device)
                spec_lengths = spec_lengths.to(device)
                wave = wave.to(device)
                wave_lengths = wave_lengths.to(device)

            use_amp = config.train.fp16_run and device.type == "cuda"

            with autocast(enabled=use_amp):
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                mel = spec_to_mel_torch(
                    spec,
                    config.data.filter_length,
                    config.data.n_mel_channels,
                    config.data.sample_rate,
                    config.data.mel_fmin,
                    config.data.mel_fmax,
                )
                y_mel = slice_segments(mel, ids_slice, config.train.segment_size // config.data.hop_length, dim=3)
                with autocast(enabled=False):
                    y_hat_mel = mel_spectrogram_torch(
                        y_hat.float().squeeze(1),
                        config.data.filter_length,
                        config.data.n_mel_channels,
                        config.data.sample_rate,
                        config.data.hop_length,
                        config.data.win_length,
                        config.data.mel_fmin,
                        config.data.mel_fmax,
                    )
                if use_amp: y_hat_mel = y_hat_mel.half()

                wave = slice_segments(wave, ids_slice * config.data.hop_length, config.train.segment_size, dim=3)
                y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())

                with autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)

            optim_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = clip_grad_value(net_d.parameters(), None)
            scaler.step(optim_d)

            with autocast(enabled=use_amp):
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
                with autocast(enabled=False):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * config.train.c_mel
                    loss_kl = (kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl)
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

                    if loss_gen_all < lowest_value["value"]:
                        lowest_value["value"] = loss_gen_all
                        lowest_value["step"] = global_step
                        lowest_value["epoch"] = epoch
                       
                        if epoch > lowest_value["epoch"]: logger.warning("CẢNH BÁO: Tổn thất tạo ra thấp hơn đã bị vượt quá tổn thất thấp hơn trong kỷ nguyên tiếp theo.")

            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = clip_grad_value(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

            if rank == 0:
                if global_step % config.train.log_interval == 0:
                    lr = optim_g.param_groups[0]["lr"]

                    if loss_mel > 75: loss_mel = 75
                    if loss_kl > 9: loss_kl = 9

                    scalar_dict = {
                        "loss/g/total": loss_gen_all,
                        "loss/d/total": loss_disc,
                        "learning_rate": lr,
                        "grad_norm_d": grad_norm_d,
                        "grad_norm_g": grad_norm_g,
                    }
                    scalar_dict.update(
                        {
                            "loss/g/fm": loss_fm,
                            "loss/g/mel": loss_mel,
                            "loss/g/kl": loss_kl,
                        }
                    )
                    scalar_dict.update(
                        {f"loss/g/{i}": v for i, v in enumerate(losses_gen)}
                    )
                    scalar_dict.update(
                        {f"loss/d_r/{i}": v for i, v in enumerate(losses_disc_r)}
                    )
                    scalar_dict.update(
                        {f"loss/d_g/{i}": v for i, v in enumerate(losses_disc_g)}
                    )
                    image_dict = {
                        "slice/mel_org": plot_spectrogram_to_numpy(
                            y_mel[0].data.cpu().numpy()
                        ),
                        "slice/mel_gen": plot_spectrogram_to_numpy(
                            y_hat_mel[0].data.cpu().numpy()
                        ),
                        "all/mel": plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                    }

                    with torch.no_grad():
                        if hasattr(net_g, "module"): o, *_ = net_g.module.infer(*reference)
                        else: o, *_ = net_g.infer(*reference)


                    audio_dict = {f"gen/audio_{global_step:07d}": o[0, :, :]}

                    summarize(
                        writer=writer,
                        global_step=global_step,
                        images=image_dict,
                        scalars=scalar_dict,
                        audios=audio_dict,
                        audio_sample_rate=config.data.sample_rate,
                    )

            global_step += 1
            pbar.update(1)

    def check_overtraining(smoothed_loss_history, threshold, epsilon=0.004):
        if len(smoothed_loss_history) < threshold + 1: return False

        for i in range(-threshold, -1):
            if smoothed_loss_history[i + 1] > smoothed_loss_history[i]: return True
            if abs(smoothed_loss_history[i + 1] - smoothed_loss_history[i]) >= epsilon: return False

        return True

    def update_exponential_moving_average(smoothed_loss_history, new_value, smoothing=0.987):
        smoothed_value = new_value if not smoothed_loss_history else (smoothing * smoothed_loss_history[-1] + (1 - smoothing) * new_value)      
        smoothed_loss_history.append(smoothed_value)
        return smoothed_value

    def save_to_json(file_path, loss_disc_history, smoothed_loss_disc_history, loss_gen_history, smoothed_loss_gen_history):
        data = {
            "loss_disc_history": loss_disc_history,
            "smoothed_loss_disc_history": smoothed_loss_disc_history,
            "loss_gen_history": loss_gen_history,
            "smoothed_loss_gen_history": smoothed_loss_gen_history,
        }

        with open(file_path, "w") as f:
            json.dump(data, f)
    
    model_add = []
    model_del = []
    done = False
    

    if rank == 0:
        if epoch % save_every_epoch == False:
            checkpoint_suffix = f"{2333333 if save_only_latest else global_step}.pth"

            save_checkpoint(net_g, optim_g, config.train.learning_rate, epoch, os.path.join(experiment_dir, "G_" + checkpoint_suffix))
            save_checkpoint(net_d, optim_d, config.train.learning_rate, epoch, os.path.join(experiment_dir, "D_" + checkpoint_suffix))

            if custom_save_every_weights: model_add.append(os.path.join(experiment_dir, f"{model_name}_{epoch}e_{global_step}s.pth"))

        if overtraining_detector and epoch > 1:
            current_loss_disc = float(loss_disc)
            loss_disc_history.append(current_loss_disc)

            smoothed_value_disc = update_exponential_moving_average(smoothed_loss_disc_history, current_loss_disc)
            is_overtraining_disc = check_overtraining(smoothed_loss_disc_history, overtraining_threshold * 2)

            if is_overtraining_disc: consecutive_increases_disc += 1
            else: consecutive_increases_disc = 0

            current_loss_gen = float(lowest_value["value"])
            loss_gen_history.append(current_loss_gen)

            smoothed_value_gen = update_exponential_moving_average(smoothed_loss_gen_history, current_loss_gen)

            is_overtraining_gen = check_overtraining(smoothed_loss_gen_history, overtraining_threshold, 0.01)

            if is_overtraining_gen: consecutive_increases_gen += 1
            else: consecutive_increases_gen = 0

            if epoch % save_every_epoch == 0: save_to_json(training_file_path, loss_disc_history, smoothed_loss_disc_history, loss_gen_history, smoothed_loss_gen_history)

            if (is_overtraining_gen and consecutive_increases_gen == overtraining_threshold or is_overtraining_disc and consecutive_increases_disc == (overtraining_threshold * 2)):
                logger.info(f"Tập luyện quá sức được phát hiện ở kỷ nguyên {epoch} với loss_g được làm mịn {smoothed_value_gen:.3f} và mất mát d {smoothed_value_disc:.3f}")
                done = True
            else:
                logger.info(f"Kỷ nguyên mới tốt nhất {epoch} với loss_g được làm mịn {smoothed_value_gen:.3f} và mất mát d {smoothed_value_disc:.3f}")

                old_model_files = glob.glob(os.path.join("assets", "weights", f"{model_name}_*e_*s_best_epoch.pth"))

                for file in old_model_files:
                    model_del.append(file)

                model_add.append(os.path.join(experiment_dir, f"{model_name}_{epoch}e_{global_step}s_best_epoch.pth"))
        
        if epoch >= custom_total_epoch:
            lowest_value_rounded = float(lowest_value["value"])
            lowest_value_rounded = round(lowest_value_rounded, 3)

            logger.info(f"Đã đào tạo hoàn thành với {epoch} kỷ nguyên, {global_step} các bước và {round(loss_gen_all.item(), 3)} mất mát gen.")
            logger.info(f"Tổn thất gen thấp nhất: {lowest_value_rounded} ở ký nguyên {lowest_value['epoch']}, bước {lowest_value['step']}")

            pid_file_path = os.path.join(experiment_dir, "config.json")

            with open(pid_file_path, "r") as pid_file:
                pid_data = json.load(pid_file)
            with open(pid_file_path, "w") as pid_file:
                pid_data.pop("process_pids", None)
                json.dump(pid_data, pid_file, indent=4)

            model_add.append(os.path.join(experiment_dir, f"{model_name}_{epoch}e_{global_step}s.pth"))
            done = True
            
        if model_add:
            ckpt = (net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict())

            for m in model_add:
                if not os.path.exists(m): extract_model(ckpt=ckpt, sr=sample_rate, pitch_guidance=pitch_guidance == True, name=model_name, model_dir=m, epoch=epoch, step=global_step, version=version, hps=hps)

        for m in model_del:
            os.remove(m)


        lowest_value_rounded = float(lowest_value["value"])
        lowest_value_rounded = round(lowest_value_rounded, 3)


        if epoch > 1 and overtraining_detector:
            remaining_epochs_gen = overtraining_threshold - consecutive_increases_gen
            remaining_epochs_disc = (overtraining_threshold * 2) - consecutive_increases_disc

            logger.info(f"{model_name} | kỷ nguyên={epoch} | bước={global_step} | {epoch_recorder.record()} | giá trị thấp nhất={lowest_value_rounded} (kỷ nguyên {lowest_value['epoch']} và bước {lowest_value['step']}) | Số kỷ nguyên còn lại để tập luyện quá sức: g/total: {remaining_epochs_gen} d/total: {remaining_epochs_disc} | làm mịn mất mát gen={smoothed_value_gen:.3f} | làm mịn mất mát disc={smoothed_value_disc:.3f}")
        elif epoch > 1 and overtraining_detector == False: logger.info(f"{model_name} | kỷ nguyên={epoch} | bước={global_step} | {epoch_recorder.record()} | giá trị thấp nhất={lowest_value_rounded} (kỷ nguyên {lowest_value['epoch']} và bước {lowest_value['step']})")
        else: logger.info(f"{model_name} | kỷ nguyên={epoch} | bước={global_step} | {epoch_recorder.record()}")

        last_loss_gen_all = loss_gen_all

        if done: os._exit(2333333)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    try:
        main()
    except Exception as e:
        logger.error(f"Đã xảy ra lỗi khi huấn luyện mô hình {e}")