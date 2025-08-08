import os
import sys
import json
import onnx
import time
import torch
import librosa
import logging
import argparse
import warnings
import onnxruntime

import numpy as np
import soundfile as sf

from tqdm import tqdm
from scipy.io import wavfile
from distutils.util import strtobool

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())

from main.inference.conversion.pipeline import Pipeline
from main.library.algorithm.synthesizers import Synthesizer
from main.app.variables import config, logger, translations
from main.library.utils import check_assets, load_audio, load_embedders_model, cut, restore, get_providers, clear_gpu_cache

for l in ["torch", "faiss", "omegaconf", "httpx", "httpcore", "faiss.loader", "numba.core", "urllib3", "transformers", "matplotlib"]:
    logging.getLogger(l).setLevel(logging.ERROR)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--convert", action='store_true')
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--filter_radius", type=int, default=3)
    parser.add_argument("--index_rate", type=float, default=0.5)
    parser.add_argument("--rms_mix_rate", type=float, default=1)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--hop_length", type=int, default=64)
    parser.add_argument("--f0_method", type=str, default="rmvpe")
    parser.add_argument("--embedder_model", type=str, default="contentvec_base")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./audios/output.wav")
    parser.add_argument("--export_format", type=str, default="wav")
    parser.add_argument("--pth_path",  type=str,  required=True)
    parser.add_argument("--index_path", type=str, default="")
    parser.add_argument("--f0_autotune", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--f0_autotune_strength", type=float, default=1)
    parser.add_argument("--clean_audio", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--clean_strength", type=float, default=0.7)
    parser.add_argument("--resample_sr", type=int, default=0)
    parser.add_argument("--split_audio", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--checkpointing", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--f0_file", type=str, default="")
    parser.add_argument("--f0_onnx", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--embedders_mode", type=str, default="fairseq")
    parser.add_argument("--formant_shifting", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--formant_qfrency", type=float, default=0.8)
    parser.add_argument("--formant_timbre", type=float, default=0.8)
    parser.add_argument("--proposal_pitch", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--proposal_pitch_threshold", type=float, default=0.0)

    return parser.parse_args()

def main():
    args = parse_arguments()
    pitch, filter_radius, index_rate, rms_mix_rate, protect, hop_length, f0_method, input_path, output_path, pth_path, index_path, f0_autotune, f0_autotune_strength, clean_audio, clean_strength, export_format, embedder_model, resample_sr, split_audio, checkpointing, f0_file, f0_onnx, embedders_mode, formant_shifting, formant_qfrency, formant_timbre, proposal_pitch, proposal_pitch_threshold = args.pitch, args.filter_radius, args.index_rate, args.rms_mix_rate,args.protect, args.hop_length, args.f0_method, args.input_path, args.output_path, args.pth_path, args.index_path, args.f0_autotune, args.f0_autotune_strength, args.clean_audio, args.clean_strength, args.export_format, args.embedder_model, args.resample_sr, args.split_audio, args.checkpointing, args.f0_file, args.f0_onnx, args.embedders_mode, args.formant_shifting, args.formant_qfrency, args.formant_timbre, args.proposal_pitch, args.proposal_pitch_threshold
    
    run_convert_script(pitch=pitch, filter_radius=filter_radius, index_rate=index_rate, rms_mix_rate=rms_mix_rate, protect=protect, hop_length=hop_length, f0_method=f0_method, input_path=input_path, output_path=output_path, pth_path=pth_path, index_path=index_path, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, clean_audio=clean_audio, clean_strength=clean_strength, export_format=export_format, embedder_model=embedder_model, resample_sr=resample_sr, split_audio=split_audio, checkpointing=checkpointing, f0_file=f0_file, f0_onnx=f0_onnx, embedders_mode=embedders_mode, formant_shifting=formant_shifting, formant_qfrency=formant_qfrency, formant_timbre=formant_timbre, proposal_pitch=proposal_pitch, proposal_pitch_threshold=proposal_pitch_threshold)

def run_convert_script(pitch=0, filter_radius=3, index_rate=0.5, rms_mix_rate=1, protect=0.5, hop_length=64, f0_method="rmvpe", input_path=None, output_path="./output.wav", pth_path=None, index_path=None, f0_autotune=False, f0_autotune_strength=1, clean_audio=False, clean_strength=0.7, export_format="wav", embedder_model="contentvec_base", resample_sr=0, split_audio=False, checkpointing=False, f0_file=None, f0_onnx=False, embedders_mode="fairseq", formant_shifting=False, formant_qfrency=0.8, formant_timbre=0.8, proposal_pitch=False, proposal_pitch_threshold=0):
    check_assets(f0_method, embedder_model, f0_onnx=f0_onnx, embedders_mode=embedders_mode)
    log_data = {translations['pitch']: pitch, translations['filter_radius']: filter_radius, translations['index_strength']: index_rate, translations['rms_mix_rate']: rms_mix_rate, translations['protect']: protect, translations['hop_length']: hop_length, translations['f0_method']: f0_method, translations['audio_path']: input_path, translations['output_path']: output_path.replace('wav', export_format), translations['model_path']: pth_path, translations['indexpath']: index_path, translations['autotune']: f0_autotune, translations['clear_audio']: clean_audio, translations['export_format']: export_format, translations['hubert_model']: embedder_model, translations['split_audio']: split_audio, translations['memory_efficient_training']: checkpointing, translations["f0_onnx_mode"]: f0_onnx, translations["embed_mode"]: embedders_mode, translations["proposal_pitch"]: proposal_pitch}

    if clean_audio: log_data[translations['clean_strength']] = clean_strength
    if resample_sr != 0: log_data[translations['sample_rate']] = resample_sr
    if f0_autotune: log_data[translations['autotune_rate_info']] = f0_autotune_strength
    if os.path.isfile(f0_file): log_data[translations['f0_file']] = f0_file
    if proposal_pitch: log_data[translations["proposal_pitch_threshold"]] = proposal_pitch_threshold
    if formant_shifting:
        log_data[translations['formant_qfrency']] = formant_qfrency
        log_data[translations['formant_timbre']] = formant_timbre

    for key, value in log_data.items():
        logger.debug(f"{key}: {value}")
    
    if not pth_path or not os.path.exists(pth_path) or os.path.isdir(pth_path) or not pth_path.endswith((".pth", ".onnx")):
        logger.warning(translations["provide_file"].format(filename=translations["model"]))
        sys.exit(1)

    cvt = VoiceConverter(pth_path, 0)
    start_time = time.time()

    pid_path = os.path.join("assets", "convert_pid.txt")
    with open(pid_path, "w") as pid_file:
        pid_file.write(str(os.getpid()))

    if os.path.isdir(input_path):
        logger.info(translations["convert_batch"])
        audio_files = [f for f in os.listdir(input_path) if f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"))]

        if not audio_files: 
            logger.warning(translations["not_found_audio"])
            sys.exit(1)

        logger.info(translations["found_audio"].format(audio_files=len(audio_files)))

        for audio in audio_files:
            audio_path = os.path.join(input_path, audio)
            output_audio = os.path.join(input_path, os.path.splitext(audio)[0] + f"_output.{export_format}")

            logger.info(f"{translations['convert_audio']} '{audio_path}'...")
            if os.path.exists(output_audio): os.remove(output_audio)

            cvt.convert_audio(pitch=pitch, filter_radius=filter_radius, index_rate=index_rate, rms_mix_rate=rms_mix_rate, protect=protect, hop_length=hop_length, f0_method=f0_method, audio_input_path=audio_path, audio_output_path=output_audio, index_path=index_path, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, clean_audio=clean_audio, clean_strength=clean_strength, export_format=export_format, embedder_model=embedder_model, resample_sr=resample_sr, checkpointing=checkpointing, f0_file=f0_file, f0_onnx=f0_onnx, embedders_mode=embedders_mode, formant_shifting=formant_shifting, formant_qfrency=formant_qfrency, formant_timbre=formant_timbre, split_audio=split_audio, proposal_pitch=proposal_pitch, proposal_pitch_threshold=proposal_pitch_threshold)

        logger.info(translations["convert_batch_success"].format(elapsed_time=f"{(time.time() - start_time):.2f}", output_path=output_path.replace('wav', export_format)))
    else:
        if not os.path.exists(input_path):
            logger.warning(translations["not_found_audio"])
            sys.exit(1)

        logger.info(f"{translations['convert_audio']} '{input_path}'...")
        if os.path.exists(output_path): os.remove(output_path)

        cvt.convert_audio(pitch=pitch, filter_radius=filter_radius, index_rate=index_rate, rms_mix_rate=rms_mix_rate, protect=protect, hop_length=hop_length, f0_method=f0_method, audio_input_path=input_path, audio_output_path=output_path, index_path=index_path, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, clean_audio=clean_audio, clean_strength=clean_strength, export_format=export_format, embedder_model=embedder_model, resample_sr=resample_sr, checkpointing=checkpointing, f0_file=f0_file, f0_onnx=f0_onnx, embedders_mode=embedders_mode, formant_shifting=formant_shifting, formant_qfrency=formant_qfrency, formant_timbre=formant_timbre, split_audio=split_audio, proposal_pitch=proposal_pitch, proposal_pitch_threshold=proposal_pitch_threshold)
        logger.info(translations["convert_audio_success"].format(input_path=input_path, elapsed_time=f"{(time.time() - start_time):.2f}", output_path=output_path.replace('wav', export_format)))

    if os.path.exists(pid_path): os.remove(pid_path)

class VoiceConverter:
    def __init__(self, model_path, sid = 0):
        self.config = config
        self.device = config.device
        self.hubert_model = None
        self.tgt_sr = None 
        self.net_g = None 
        self.vc = None
        self.cpt = None  
        self.version = None 
        self.n_spk = None  
        self.use_f0 = None  
        self.loaded_model = None
        self.vocoder = "Default"
        self.checkpointing = False
        self.sample_rate = 16000
        self.sid = sid
        self.get_vc(model_path, sid)

    def convert_audio(self, audio_input_path, audio_output_path, index_path, embedder_model, pitch, f0_method, index_rate, rms_mix_rate, protect, hop_length, f0_autotune, f0_autotune_strength, filter_radius, clean_audio, clean_strength, export_format, resample_sr = 0, checkpointing = False, f0_file = None, f0_onnx = False, embedders_mode = "fairseq", formant_shifting = False, formant_qfrency = 0.8, formant_timbre = 0.8, split_audio = False, proposal_pitch = False, proposal_pitch_threshold = 0):
        try:
            with tqdm(total=10, desc=translations["convert_audio"], ncols=100, unit="a", leave=not split_audio) as pbar:
                audio = load_audio(audio_input_path, self.sample_rate, formant_shifting=formant_shifting, formant_qfrency=formant_qfrency, formant_timbre=formant_timbre)
                self.checkpointing = checkpointing

                audio_max = np.abs(audio).max() / 0.95
                if audio_max > 1: audio /= audio_max

                if not self.hubert_model:
                    models, embed_suffix = load_embedders_model(embedder_model, embedders_mode)
                    if embed_suffix in [".pt", ".safetensors"]: models = (models.half() if self.config.is_half else models.float()).eval().to(self.device)

                    self.hubert_model = models
                    self.embed_suffix = embed_suffix

                pbar.update(1)
                if split_audio:
                    pbar.close()
                    chunks = cut(audio, self.sample_rate, db_thresh=-60, min_interval=500)  

                    logger.info(f"{translations['split_total']}: {len(chunks)}")
                    pbar = tqdm(total=len(chunks) * 5 + 4, desc=translations["convert_audio"], ncols=100, unit="a", leave=True)
                else: chunks = [(audio, 0, 0)]

                pbar.update(1)
                converted_chunks = [(
                    start, 
                    end, 
                    self.vc.pipeline(
                        logger=logger, 
                        model=self.hubert_model, 
                        net_g=self.net_g, 
                        sid=self.sid, 
                        audio=waveform, 
                        f0_up_key=pitch, 
                        f0_method=f0_method, 
                        file_index=(index_path.strip().strip('"').strip("\n").strip('"').strip().replace("trained", "added")), 
                        index_rate=index_rate, 
                        pitch_guidance=self.use_f0, 
                        filter_radius=filter_radius, 
                        rms_mix_rate=rms_mix_rate, 
                        version=self.version, 
                        protect=protect, 
                        hop_length=hop_length, 
                        f0_autotune=f0_autotune, 
                        f0_autotune_strength=f0_autotune_strength, 
                        suffix=self.suffix, 
                        embed_suffix=self.embed_suffix, 
                        f0_file=f0_file, 
                        f0_onnx=f0_onnx, 
                        pbar=pbar, 
                        proposal_pitch=proposal_pitch,
                        proposal_pitch_threshold=proposal_pitch_threshold,
                        energy_use=self.energy,
                        del_onnx=not split_audio
                    )
                ) for waveform, start, end in chunks]

                pbar.update(1)

                del self.net_g, self.hubert_model
                audio_output = restore(converted_chunks, total_len=len(audio), dtype=converted_chunks[0][2].dtype) if split_audio else converted_chunks[0][2]
                
                if self.tgt_sr != resample_sr and resample_sr > 0: 
                    audio_output = librosa.resample(audio_output, orig_sr=self.tgt_sr, target_sr=resample_sr, res_type="soxr_vhq")
                    self.tgt_sr = resample_sr

                pbar.update(1)
                if clean_audio:
                    from main.tools.noisereduce import reduce_noise
                    audio_output = reduce_noise(y=audio_output, sr=self.tgt_sr, prop_decrease=clean_strength, device=self.device) 

                if len(audio) / self.sample_rate > len(audio_output) / self.tgt_sr:
                    padding = np.zeros(int(np.round(len(audio) / self.sample_rate * self.tgt_sr) - len(audio_output)), dtype=audio_output.dtype)
                    audio_output = np.concatenate([audio_output, padding])

                try:
                    if export_format.endswith("wav"): wavfile.write(audio_output_path, self.tgt_sr, audio_output.astype(np.float32))
                    else: sf.write(audio_output_path, audio_output, self.tgt_sr, format=export_format)
                except:
                    sf.write(audio_output_path, librosa.resample(audio_output, orig_sr=self.tgt_sr, target_sr=48000, res_type="soxr_vhq"), 48000, format=export_format)

                pbar.update(1)
        except Exception as e:
            logger.error(translations["error_convert"].format(e=e))
            import traceback
            logger.debug(traceback.format_exc())

    def get_vc(self, weight_root, sid):
        if sid == "" or sid == []:
            self.cleanup()
            clear_gpu_cache()

        if not self.loaded_model or self.loaded_model != weight_root:
            self.loaded_model = weight_root
            self.load_model()
            if self.cpt is not None: self.setup()

    def cleanup(self):
        if self.hubert_model is not None:
            del self.net_g, self.n_spk, self.vc, self.hubert_model, self.tgt_sr
            self.hubert_model = self.net_g = self.n_spk = self.vc = self.tgt_sr = None
            clear_gpu_cache()

        del self.net_g, self.cpt
        clear_gpu_cache()
        self.cpt = None

    def load_model(self):
        if os.path.isfile(self.loaded_model):
            if self.loaded_model.endswith(".pth"): self.cpt = torch.load(self.loaded_model, map_location="cpu", weights_only=True)  
            else: 
                sess_options = onnxruntime.SessionOptions()
                sess_options.log_severity_level = 3
                self.cpt = onnxruntime.InferenceSession(self.loaded_model, sess_options=sess_options, providers=get_providers())
        else: self.cpt = None

    def setup(self):
        if self.cpt is not None:
            if self.loaded_model.endswith(".pth"):
                self.tgt_sr = self.cpt["config"][-1]
                self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]

                self.use_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", "v1")
                self.vocoder = self.cpt.get("vocoder", "Default")
                self.energy = self.cpt.get("energy", False)

                if self.vocoder != "Default": self.config.is_half = False
                self.net_g = Synthesizer(*self.cpt["config"], use_f0=self.use_f0, text_enc_hidden_dim=768 if self.version == "v2" else 256, vocoder=self.vocoder, checkpointing=self.checkpointing, energy=self.energy)
                del self.net_g.enc_q

                self.net_g.load_state_dict(self.cpt["weight"], strict=False)
                self.net_g.eval().to(self.device)
                self.net_g = (self.net_g.half() if self.config.is_half else self.net_g.float())
                self.n_spk = self.cpt["config"][-3]
                self.suffix = ".pth"
            else:
                metadata_dict = None
                for prop in onnx.load(self.loaded_model).metadata_props:
                    if prop.key == "model_info":
                        metadata_dict = json.loads(prop.value)
                        break

                self.net_g = self.cpt
                self.tgt_sr = metadata_dict.get("sr", 32000)
                self.use_f0 = metadata_dict.get("f0", 1)
                self.version = metadata_dict.get("version", "v1")
                self.energy = metadata_dict.get("energy", False)
                self.suffix = ".onnx"

            self.vc = Pipeline(self.tgt_sr, self.config)

if __name__ == "__main__": main()