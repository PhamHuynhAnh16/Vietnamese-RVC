import os
import gc
import sys
import time
import tqdm
import torch
import traceback
import concurrent.futures

import numpy as np

sys.path.append(os.getcwd())

from main.app.variables import logger, translations, config
from main.inference.extracting.setup_path import setup_paths
from main.library.utils import load_audio, load_embedders_model, extract_features

def process_file_embedding(files, embedder_model, embedders_mode, device, version, is_half, threads):
    model, embed_suffix = load_embedders_model(embedder_model, embedders_mode)
    if embed_suffix != ".onnx": model = model.to(device).to(torch.float16 if is_half else torch.float32).eval()
    threads = max(1, threads)

    def worker(file_info):
        try:
            file, out_path = file_info
            out_file_path = os.path.join(out_path, os.path.basename(file.replace("wav", "npy"))) if os.path.isdir(out_path) else out_path
            if os.path.exists(out_file_path): return
            feats = torch.from_numpy(load_audio(file, 16000)).to(device).to(torch.float16 if is_half else torch.float32).view(1, -1)

            with torch.no_grad():
                if embed_suffix == ".pt":
                    logits = model.extract_features(**{"source": feats, "padding_mask": torch.BoolTensor(feats.shape).fill_(False).to(device), "output_layer": 9 if version == "v1" else 12})
                    feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
                elif embed_suffix == ".onnx": feats = extract_features(model, feats, version).to(device)
                elif embed_suffix == ".safetensors":
                    logits = model(feats)["last_hidden_state"]
                    feats = (model.final_proj(logits[0]).unsqueeze(0) if version == "v1" else logits)
                else: raise ValueError(translations["option_not_valid"])

            feats = feats.squeeze(0).float().cpu().numpy()
            if not np.isnan(feats).any(): np.save(out_file_path, feats, allow_pickle=False)
            else: logger.warning(f"{file} {translations['NaN']}")
        except:
            logger.debug(traceback.format_exc())

    with tqdm.tqdm(total=len(files), ncols=100, unit="p", leave=True) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for _ in concurrent.futures.as_completed([executor.submit(worker, f) for f in files]):
                pbar.update(1)

def run_embedding_extraction(exp_dir, version, num_processes, devices, embedder_model, embedders_mode, is_half):
    wav_path, out_path = setup_paths(exp_dir, version)
    start_time = time.time()

    logger.info(translations["start_extract_hubert"])
    num_processes = 1 if config.device.startswith("ocl") and embedders_mode == "onnx" else num_processes
    paths = sorted([(os.path.join(wav_path, file), out_path) for file in os.listdir(wav_path) if file.endswith(".wav")])

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        concurrent.futures.wait([executor.submit(process_file_embedding, paths[i::len(devices)], embedder_model, embedders_mode, devices[i], version, is_half, num_processes // len(devices)) for i in range(len(devices))])
    
    gc.collect()
    logger.info(translations["extract_hubert_success"].format(elapsed_time=f"{(time.time() - start_time):.2f}"))

def create_mute_file(version, embedder_model, embedders_mode, is_half):
    start_time = time.time()
    logger.info(translations["start_extract_hubert"])

    process_file_embedding([(os.path.join("assets", "logs", "mute", "sliced_audios_16k", "mute.wav"), os.path.join("assets", "logs", "mute", f"{version}_extracted", f"mute_{embedder_model.replace('_hubert_base', '')}.npy"))], embedder_model, embedders_mode, config.device, version, is_half, 1)

    gc.collect()
    logger.info(translations["extract_hubert_success"].format(elapsed_time=f"{(time.time() - start_time):.2f}"))