import os
import sys
import time
import torch
import librosa
import logging
import argparse
import warnings

import numpy as np
import soundfile as sf

from tqdm import tqdm
from scipy import signal

sys.path.append(os.getcwd())

from main.library.audio.upscaler import FlashSR
from main.app.core.ui import replace_export_format
from main.library.audio.noisereduce import TorchGate
from main.inference.conversion.pipeline import Pipeline
from main.library.audio.audio import load_audio, cut, restore
from main.app.variables import config, logger, translations, file_types
from main.library.audio.audio_processing import preprocess, postprocess
from main.library.utils import check_assets, check_upscaler, load_embedders_model, load_model, strtobool, load_faiss_index

if not config.debug_mode:
    warnings.filterwarnings("ignore")

    for l in [
        "torch", 
        "faiss", 
        "omegaconf", 
        "httpx", 
        "httpcore", 
        "faiss.loader", 
        "numba.core", 
        "urllib3", 
        "transformers", 
        "matplotlib"
    ]:
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
    parser.add_argument("--embedder_model", type=str, default="hubert_base")
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
    parser.add_argument("--predictor_onnx", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--embedders_mode", type=str, default="fairseq")
    parser.add_argument("--formant_shifting", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--formant_qfrency", type=float, default=0.8)
    parser.add_argument("--formant_timbre", type=float, default=0.8)
    parser.add_argument("--proposal_pitch", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--proposal_pitch_threshold", type=float, default=255.0)
    parser.add_argument("--audio_processing", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--sid", type=int, default=0, required=False)
    parser.add_argument("--embedders_mix", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--embedders_mix_layers", type=int, default=9, required=False)
    parser.add_argument("--embedders_mix_ratio", type=float, default=0.5)
    parser.add_argument("--noise_scale", type=float, default=0.35)
    parser.add_argument("--nprobe", type=int, default=1)
    parser.add_argument("--audio_upscaler", type=lambda x: bool(strtobool(x)), default=False)

    return parser.parse_args()

def main():
    args = parse_arguments()

    (
        pitch, 
        filter_radius, 
        index_rate, 
        rms_mix_rate, 
        protect, 
        hop_length, 
        f0_method, 
        input_path, 
        output_path, 
        pth_path, 
        index_path, 
        f0_autotune, 
        f0_autotune_strength, 
        clean_audio, 
        clean_strength, 
        export_format, 
        embedder_model, 
        resample_sr, 
        split_audio, 
        checkpointing, 
        f0_file, 
        predictor_onnx, 
        embedders_mode, 
        formant_shifting, 
        formant_qfrency, 
        formant_timbre, 
        proposal_pitch, 
        proposal_pitch_threshold, 
        audio_processing, 
        alpha,
        sid,
        embedders_mix,
        embedders_mix_layers,
        embedders_mix_ratio,
        noise_scale,
        nprobe,
        audio_upscaler
    ) = (
        args.pitch, 
        args.filter_radius, 
        args.index_rate, 
        args.rms_mix_rate,
        args.protect, 
        args.hop_length, 
        args.f0_method, 
        args.input_path, 
        args.output_path, 
        args.pth_path, 
        args.index_path, 
        args.f0_autotune, 
        args.f0_autotune_strength, 
        args.clean_audio, 
        args.clean_strength, 
        args.export_format, 
        args.embedder_model, 
        args.resample_sr, 
        args.split_audio, 
        args.checkpointing, 
        args.f0_file, 
        args.predictor_onnx, 
        args.embedders_mode, 
        args.formant_shifting, 
        args.formant_qfrency, 
        args.formant_timbre, 
        args.proposal_pitch, 
        args.proposal_pitch_threshold, 
        args.audio_processing, 
        args.alpha,
        args.sid,
        args.embedders_mix,
        args.embedders_mix_layers,
        args.embedders_mix_ratio,
        args.noise_scale,
        args.nprobe,
        args.audio_upscaler
    )
    
    run_convert_script(
        pitch=pitch, 
        filter_radius=filter_radius, 
        index_rate=index_rate, 
        rms_mix_rate=rms_mix_rate, 
        protect=protect, 
        hop_length=hop_length, 
        f0_method=f0_method, 
        input_path=input_path, 
        output_path=output_path, 
        pth_path=pth_path, 
        index_path=index_path, 
        f0_autotune=f0_autotune, 
        f0_autotune_strength=f0_autotune_strength, 
        clean_audio=clean_audio, 
        clean_strength=clean_strength, 
        export_format=export_format, 
        embedder_model=embedder_model, 
        resample_sr=resample_sr, 
        split_audio=split_audio, 
        checkpointing=checkpointing, 
        f0_file=f0_file, 
        predictor_onnx=predictor_onnx, 
        embedders_mode=embedders_mode, 
        formant_shifting=formant_shifting, 
        formant_qfrency=formant_qfrency, 
        formant_timbre=formant_timbre, 
        proposal_pitch=proposal_pitch, 
        proposal_pitch_threshold=proposal_pitch_threshold, 
        audio_processing=audio_processing, 
        alpha=alpha,
        sid=sid,
        embedders_mix=embedders_mix, 
        embedders_mix_layers=embedders_mix_layers, 
        embedders_mix_ratio=embedders_mix_ratio,
        noise_scale=noise_scale,
        nprobe=nprobe,
        audio_upscaler=audio_upscaler
    )

def run_convert_script(
    pitch=0, 
    filter_radius=3, 
    index_rate=0.5, 
    rms_mix_rate=1, 
    protect=0.5, 
    hop_length=64, 
    f0_method="rmvpe", 
    input_path=None, 
    output_path="./output.wav", 
    pth_path=None, 
    index_path=None, 
    f0_autotune=False, 
    f0_autotune_strength=1, 
    clean_audio=False, 
    clean_strength=0.7, 
    export_format="wav", 
    embedder_model="hubert_base", 
    resample_sr=0, 
    split_audio=False, 
    checkpointing=False, 
    f0_file=None, 
    predictor_onnx=False, 
    embedders_mode="fairseq", 
    formant_shifting=False, 
    formant_qfrency=0.8, 
    formant_timbre=0.8, 
    proposal_pitch=False, 
    proposal_pitch_threshold=255.0, 
    audio_processing=False,
    alpha=0.5,
    sid=0,
    embedders_mix = False,
    embedders_mix_layers = 9,
    embedders_mix_ratio = 0.5,
    noise_scale = 0.35,
    nprobe = 1,
    audio_upscaler = False
):
    if audio_upscaler: check_upscaler()
    check_assets(f0_method, embedder_model, predictor_onnx=predictor_onnx, embedders_mode=embedders_mode)

    log_data = {
        translations["pitch"]: pitch, 
        translations["filter_radius"]: filter_radius, 
        translations["index_strength"]: index_rate, 
        translations["rms_mix_rate"]: rms_mix_rate, 
        translations["protect"]: protect, 
        translations["hop_length"]: hop_length, 
        translations["f0_method"]: f0_method, 
        translations["audio_path"]: input_path, 
        translations["output_path"]: replace_export_format(output_path, export_format), 
        translations["model_path"]: pth_path, 
        translations["indexpath"]: index_path, 
        translations["autotune"]: f0_autotune, 
        translations["autotune_rate_info"]: f0_autotune_strength,
        translations["clear_audio"]: clean_audio, 
        translations["clean_strength"]: clean_strength,
        translations["sample_rate"]: resample_sr,
        translations["export_format"]: export_format, 
        translations["hubert_model"]: embedder_model, 
        translations["split_audio"]: split_audio, 
        translations["memory_efficient_training"]: checkpointing, 
        translations["predictor_onnx"]: predictor_onnx, 
        translations["f0_file"]: f0_file,
        translations["embed_mode"]: embedders_mode, 
        translations["proposal_pitch"]: proposal_pitch, 
        translations["proposal_pitch_threshold"]: proposal_pitch_threshold,
        translations["audio_processing"]: audio_processing,
        translations["alpha_label"]: alpha,
        translations["embedders_mix"]: embedders_mix,
        translations["embedders_mix_layers"]: embedders_mix_layers,
        translations["embedders_mix_ratio"]: embedders_mix_ratio,
        translations["formant_qfrency"]: formant_qfrency,
        translations["formant_timbre"]: formant_timbre,
        translations["noise_scale"]: noise_scale,
        translations["nprobe"]: nprobe,
        translations["audio_upscaler"]: audio_upscaler
    }

    for key, value in log_data.items():
        logger.debug(f"{key}: {value}")
    
    if not pth_path or not os.path.exists(pth_path) or os.path.isdir(pth_path) or not pth_path.endswith((".pth", ".onnx")):
        logger.warning(translations["provide_file"].format(filename=translations["model"]))
        sys.exit(1)

    cvt = VoiceConverter(pth_path, embedder_model, embedders_mode, sid, noise_scale, checkpointing, hop_length, alpha, predictor_onnx, clean_audio, clean_strength, audio_upscaler)

    pid_path = os.path.join("assets", "convert_pid.txt")
    with open(pid_path, "w") as pid_file:
        pid_file.write(str(os.getpid()))

    def convert_audio(audio_path, output_audio):
        cvt.convert_audio(
            pitch=pitch, 
            filter_radius=filter_radius, 
            index_rate=index_rate, 
            rms_mix_rate=rms_mix_rate, 
            protect=protect, 
            f0_method=f0_method, 
            audio_input_path=audio_path, 
            audio_output_path=output_audio, 
            index_path=index_path, 
            f0_autotune=f0_autotune, 
            f0_autotune_strength=f0_autotune_strength, 
            export_format=export_format, 
            resample_sr=resample_sr, 
            f0_file=f0_file, 
            formant_shifting=formant_shifting, 
            formant_qfrency=formant_qfrency, 
            formant_timbre=formant_timbre, 
            split_audio=split_audio, 
            proposal_pitch=proposal_pitch, 
            proposal_pitch_threshold=proposal_pitch_threshold,
            audio_processing=audio_processing,
            embedders_mix=embedders_mix, 
            embedders_mix_layers=embedders_mix_layers, 
            embedders_mix_ratio=embedders_mix_ratio,
            nprobe=nprobe
        )
    
    start_time = time.time()

    if os.path.isdir(input_path):
        logger.info(translations["convert_batch"])

        audio_files = [
            f 
            for f in os.listdir(input_path) 
            if f.lower().endswith(tuple(file_types))
        ]

        if not audio_files: 
            logger.warning(translations["not_found_audio"])
            sys.exit(1)

        logger.info(translations["found_audio"].format(audio_files=len(audio_files)))

        for audio in audio_files:
            audio_path = os.path.join(input_path, audio)
            output_audio = os.path.join(input_path, os.path.splitext(audio)[0] + f"_output.{export_format}")

            logger.info(f"{translations['convert_audio']} '{audio_path}'...")
            if os.path.exists(output_audio): os.remove(output_audio)

            convert_audio(audio_path, output_audio)

        logger.info(
            translations["convert_batch_success"].format(
                elapsed_time=f"{(time.time() - start_time):.2f}", 
                output_path=replace_export_format(output_path, export_format)
            )
        )
    else:
        if not os.path.exists(input_path):
            logger.warning(translations["not_found_audio"])
            sys.exit(1)

        logger.info(f"{translations['convert_audio']} '{input_path}'...")
        if os.path.exists(output_path): os.remove(output_path)

        convert_audio(input_path, output_path)

        logger.info(
            translations["convert_audio_success"].format(
                input_path=input_path, 
                elapsed_time=f"{(time.time() - start_time):.2f}", 
                output_path=replace_export_format(output_path, export_format)
            )
        )

    if os.path.exists(pid_path): os.remove(pid_path)

class VoiceConverter:
    def __init__(
        self, 
        model_path, 
        embedder_model, 
        embedders_mode,
        sid = 0,
        noise_scale = 0.35,
        checkpointing = False,
        hop_length = 160,
        alpha = 0.5,
        predictor_onnx = False,
        clean_audio = False,
        clean_strength = 0.5,
        audio_upscaler = False
    ):
        self.vc = None
        self.index = None
        self.net_g = None 
        self.tgt_sr = None 
        self.big_tsr = None
        self.rms_extract = None
        self.hubert_model = None
        self.f0_generator = None

        self.alpha = alpha
        self.sample_rate = 16000
        self.hop_length = hop_length
        self.predictor_onnx = predictor_onnx
        self.dtype = torch.float16 if config.is_half else torch.float32

        self.setup_hubert(embedder_model, embedders_mode)
        self.setup_vc(model_path, sid, checkpointing, noise_scale)
        self.tg = TorchGate(self.tgt_sr, prop_decrease=clean_strength).to(config.device) if clean_audio else None
        self.flash_sr = FlashSR(os.path.join("assets", "models", "upscalers", "upscalers.pth"), device=config.device, is_half=config.is_half) if audio_upscaler else None

    def convert_audio(
        self, 
        audio_input_path, 
        audio_output_path, 
        index_path, 
        pitch, 
        f0_method, 
        index_rate, 
        rms_mix_rate, 
        protect,  
        f0_autotune, 
        f0_autotune_strength, 
        filter_radius, 
        export_format, 
        resample_sr = 0, 
        f0_file = None, 
        formant_shifting = False, 
        formant_qfrency = 0.8, 
        formant_timbre = 0.8, 
        split_audio = False, 
        proposal_pitch = False, 
        proposal_pitch_threshold = 0, 
        audio_processing = False, 
        embedders_mix = False,
        embedders_mix_layers = 9,
        embedders_mix_ratio = 0.5,
        nprobe = 1
    ):
        try:
            with tqdm(total=10, desc=translations["convert_audio"], ncols=100, unit="a", leave=not split_audio) as pbar:
                audio = load_audio(audio_input_path, sample_rate=self.sample_rate, formant_shifting=formant_shifting, formant_qfrency=formant_qfrency, formant_timbre=formant_timbre)
                if audio_processing: audio = preprocess(audio, self.sample_rate)

                try:
                    audio_max = np.abs(audio).max() / 0.95
                    if audio_max > 1: audio /= audio_max
                except:
                    import shutil

                    shutil.copy(audio_input_path, audio_output_path)
                    return
                
                if index_rate != 0 and (self.index is None or self.big_tsr is None):
                    self.index, self.big_tsr = load_faiss_index(index_path.strip().strip('"').strip("\n").strip('"').strip().replace("trained", "added"), nprobe)
                    if len(audio) > 10e6: self.index.search = self.index._search_cpu

                pbar.update(1)
                if split_audio:
                    pbar.close()
                    chunks = cut(audio, self.sample_rate, db_thresh=-60, min_interval=500)

                    logger.info(f"{translations['split_total']}: {len(chunks)}")
                    pbar = tqdm(total=len(chunks) * 5 + 4, desc=translations["convert_audio"], ncols=100, unit="a", leave=True)
                else: chunks = [(audio, 0, 0)]

                pbar.update(1)
                converted_chunks = [
                    (
                        start, 
                        end, 
                        self.vc.pipeline(
                            model=self.hubert_model, 
                            net_g=self.net_g, 
                            audio=waveform, 
                            f0_up_key=pitch, 
                            f0_method=f0_method, 
                            index=self.index,
                            big_tsr=self.big_tsr, 
                            index_rate=index_rate, 
                            filter_radius=filter_radius, 
                            rms_mix_rate=rms_mix_rate, 
                            protect=protect, 
                            f0_autotune=f0_autotune, 
                            f0_autotune_strength=f0_autotune_strength, 
                            f0_file=f0_file, 
                            pbar=pbar, 
                            proposal_pitch=proposal_pitch,
                            proposal_pitch_threshold=proposal_pitch_threshold,
                            embedders_mix=embedders_mix, 
                            embedders_mix_layers=embedders_mix_layers, 
                            embedders_mix_ratio=embedders_mix_ratio,
                        )
                    ) for waveform, start, end in chunks
                ]

                pbar.update(1)
                audio_output = restore(converted_chunks, total_len=len(audio), dtype=converted_chunks[0][2].dtype) if split_audio else converted_chunks[0][2]

                if audio_processing: audio_output = postprocess(audio_output, self.tgt_sr)
                if self.tg is not None: audio_output = self.tg(torch.from_numpy(audio_output).unsqueeze(0).to(config.device).float()).squeeze(0).cpu().detach().numpy()

                audio_output_resample = None
                target_len = int(np.round(len(audio) / self.sample_rate * self.tgt_sr))
                if len(audio_output) != target_len: audio_output = signal.resample_poly(audio_output, target_len, len(audio_output))

                if self.flash_sr is not None:
                    audio_output_resample = self.flash_sr.upscaler(audio_output, sample_rate=self.tgt_sr, pbar=pbar)
                    self.tgt_sr = 192000
                elif self.tgt_sr != resample_sr and resample_sr > 0: 
                    audio_output_resample = librosa.resample(audio_output, orig_sr=self.tgt_sr, target_sr=resample_sr, res_type="soxr_vhq")
                    self.tgt_sr = resample_sr

                pbar.update(1)

                try:
                    sf.write(
                        audio_output_path, 
                        audio_output if audio_output_resample is None else audio_output_resample, 
                        self.tgt_sr, 
                        format=export_format
                    )
                except:
                    logger.info(translations["sr_not_support"])

                    sf.write(
                        audio_output_path, 
                        librosa.resample(audio_output, orig_sr=self.tgt_sr, target_sr=48000, res_type="soxr_vhq"), 
                        48000, 
                        format=export_format
                    )

                pbar.update(1)
        except Exception as e:
            import traceback

            logger.debug(traceback.format_exc())
            logger.error(translations["error_convert"].format(e=e))
    
    def setup_hubert(self, embedder_model, embedders_mode):
        models = load_embedders_model(embedder_model, embedders_mode)

        if isinstance(models, torch.nn.Module): 
            models = models.to(config.device).to(self.dtype).eval()
            if config.compile_all and embedders_mode != "whisper": models = torch.compile(models, mode=config.compile_mode)

        self.hubert_model = models
    
    def setup_predictor(self):
        from main.library.predictors.Generator import Generator

        self.f0_generator = Generator(
            self.sample_rate, 
            self.hop_length, 
            config.configs.get("f0_min", 50), 
            config.configs.get("f0_max", 1100), 
            self.alpha, 
            config.is_half, 
            config.device, 
            self.predictor_onnx
        )
    
    def setup_rms(self):
        from main.inference.extracting.rms import RMSEnergyExtractor

        self.rms_extract = RMSEnergyExtractor(
            frame_length=2048, 
            hop_length=self.sample_rate // 100, 
            center=True, 
            pad_mode = "reflect"
        ).to(config.device).eval()

    def setup_vc(self, weight_root, sid, checkpointing, noise_scale):
        model = load_model(weight_root)

        if weight_root.endswith(".pth"):
            from main.library.algorithm.synthesizers import Synthesizer, SynthesizerSVC

            self.tgt_sr = model["config"][-1]
            model["config"][-3] = model["weight"]["emb_g.weight"].shape[0]

            use_f0 = model.get("f0", 1)
            version = model.get("version", "v1")
            energy = model.get("energy", False)
            vocoder = model.get("vocoder", "Default")
            hidden_dim = 768 if version == "v2" else 256

            if model.get("architecture", "RVC"):
                self.net_g = Synthesizer(
                    *model["config"], 
                    use_f0=use_f0, 
                    text_enc_hidden_dim=hidden_dim, 
                    vocoder=vocoder, 
                    checkpointing=checkpointing, 
                    energy=energy
                )
            else:
                self.net_g = SynthesizerSVC(
                    *model["config"], 
                    text_enc_hidden_dim=hidden_dim, 
                    vocoder=vocoder, 
                    checkpointing=checkpointing, 
                    noise_scale=noise_scale
                )

            del self.net_g.enc_q
            self.net_g.load_state_dict(model["weight"], strict=False)
            self.net_g.eval().to(config.device).to(self.dtype)
            if config.compile_all: self.net_g = torch.compile(self.net_g, mode=config.compile_mode)
        else:
            self.net_g = model.to(config.device)
            self.tgt_sr = model.cpt.get("tgt_sr", 32000)

            use_f0 = model.cpt.get("f0", 1)
            version = model.cpt.get("version", "v1")
            energy = model.cpt.get("energy", False)

        if energy: self.setup_rms()
        if use_f0: self.setup_predictor()

        sid = torch.tensor(sid, device=config.device).unsqueeze(0).long()
        self.vc = Pipeline(self.tgt_sr, config, self.f0_generator, self.rms_extract, version, sid, self.dtype)

if __name__ == "__main__": main()