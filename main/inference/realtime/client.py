import os
import gc
import sys
import json

import numpy as np

from fastapi import FastAPI, WebSocketDisconnect, WebSocket, Request

sys.path.append(os.getcwd())

from main.library.utils import clear_gpu_cache
from main.inference.realtime.realtime import VoiceChanger
from main.app.variables import configs, translations, logger, config

app = FastAPI()
vc_instance = None
params = {}

PIPELINE_SAMPLE_RATE = 16000
DEVICE_SAMPLE_RATE = 48000

@app.websocket("/change-config")
async def change_config(ws: WebSocket):
    """
    Handles hot-reloading and runtime reconfiguration of the voice conversion model parameters 
    such as crossfade boundaries, VAD modes, audio cleaners, effects, and index paths via a persistent WebSocket.

    Args:
        ws (WebSocket): The incoming communication handshake instance from the client layer.
    """

    global vc_instance, params

    await ws.accept()
    if vc_instance is None: return

    text = await ws.receive_text()
    jsons = json.loads(text)

    # Distinguish standard execution settings from extra dictionary keyword arguments (kwargs)
    if jsons["if_kwargs"] and jsons["value"] is not None:
        params["kwargs"][jsons["key"]] = jsons["value"]
    elif jsons["value"] is not None:
        params[jsons["key"]] = jsons["value"]

    # Calculate overlapping structure size windows mapped against the target system sample rate
    crossfade_frame = int(params.get("cross_fade_overlap_size", 0.1) * DEVICE_SAMPLE_RATE)
    extra_frame = int(params.get("extra_convert_size", 0.5) * DEVICE_SAMPLE_RATE)

    # Trigger memory reallocation on the underlying core if latency window sizes are altered
    if (
        vc_instance.crossfade_frame != crossfade_frame or
        vc_instance.extra_frame != extra_frame
    ):
        # Force-release old arrays to prevent internal calculation leaks before reconfiguration
        del (
            vc_instance.fade_in_window,
            vc_instance.fade_out_window,
            vc_instance.sola_buffer
        )

        vc_instance.vc_model.realloc(
            vc_instance.block_frame,
            vc_instance.extra_frame,
            vc_instance.crossfade_frame,
            vc_instance.sola_search_frame,
        )
        vc_instance.generate_strength()

    # Convert the dB floor threshold configuration parameter to standard linear amplitude values
    vc_instance.vc_model.input_sensitivity = 10 ** (params.get("silent_threshold", -90) / 20)

    # Voice Activity Detection (VAD) Engine Configuration Block
    vad_enabled = params.get("vad_enabled", True)
    sensitivity_mode = params.get("vad_sensitivity", 3)
    vad_frame_ms = params.get("vad_frame_ms", 30)

    if vad_enabled is False:
        vc_instance.vc_model.vad = None
    elif vad_enabled and vc_instance.vc_model.vad is None:
        from main.inference.realtime.vad_utils import VADProcessor

        vc_instance.vc_model.vad = VADProcessor(
            sensitivity_mode=sensitivity_mode,
            sample_rate=vc_instance.vc_model.sample_rate,
            frame_duration_ms=vad_frame_ms
        )

    if vc_instance.vc_model.vad is not None:
        vc_instance.vc_model.vad.vad.set_mode(sensitivity_mode)
        vc_instance.vc_model.vad.frame_length = int(vc_instance.vc_model.sample_rate * (vad_frame_ms / 1000.0))

    # Audio Clean Gate (TorchGate Noise Reduction) Block
    clean_audio = params.get("clean_audio", False)
    clean_strength = params.get("clean_strength", 0.5)

    if clean_audio is False:
        vc_instance.vc_model.tg = None
    elif clean_audio and vc_instance.vc_model.tg is None:
        from main.library.audio.noisereduce import TorchGate

        vc_instance.vc_model.tg = (
            TorchGate(
                vc_instance.vc_model.pipeline.tgt_sr,
                prop_decrease=clean_strength,
            ).to(config.device)
        )

    if vc_instance.vc_model.tg is not None:
        vc_instance.vc_model.tg.prop_decrease = clean_strength

    # Pedalboard Effects Post-Processing Block
    post_process = params.get("post_process", False)
    kwargs = params.get("kwargs", {})

    if post_process is False:
        vc_instance.vc_model.board = None
        vc_instance.vc_model.kwargs = None
    elif post_process and vc_instance.vc_model.kwargs != kwargs:
        new_board = vc_instance.vc_model.setup_pedalboard(**kwargs)
        vc_instance.vc_model.board = new_board
        vc_instance.vc_model.kwargs = kwargs.copy()

    # Model Weight Reload / Architecture Swap Block
    noise_scale = params.get("noise_scale", 0.35)
    model_pth = params.get("model_path", vc_instance.vc_model.model_path)
    model_pth = os.path.join(configs["weights_path"], model_pth) if not os.path.exists(model_pth) else model_pth

    if model_pth and vc_instance.vc_model.model_path != model_pth:
        import torch
        import torchaudio.transforms as tat

        vc_instance.vc_model.model_path = model_pth
        vc_instance.vc_model.pipeline.vc.setup(model_pth, noise_scale)
        vc_instance.vc_model.pipeline.use_f0 = vc_instance.vc_model.pipeline.vc.use_f0
        vc_instance.vc_model.pipeline.tgt_sr = vc_instance.vc_model.pipeline.vc.tgt_sr
        vc_instance.vc_model.pipeline.version = vc_instance.vc_model.pipeline.vc.version

        # Re-initialize the resampling layer since the target sample rate might have changed
        vc_instance.vc_model.resample_out = tat.Resample(
            orig_freq=vc_instance.vc_model.pipeline.tgt_sr,
            new_freq=vc_instance.vc_model.output_sample_rate,
            dtype=torch.float32
        ).to(config.device)

        if clean_audio:
            from main.library.audio.noisereduce import TorchGate

            vc_instance.vc_model.tg = (
                TorchGate(
                    vc_instance.vc_model.pipeline.tgt_sr,
                    prop_decrease=clean_strength,
                ).to(config.device)
            )

    # Speaker Identity ID Configuration
    sid = params.get("sid", vc_instance.vc_model.pipeline.sid)
    if vc_instance.vc_model.pipeline.sid != sid:
        import torch
        vc_instance.vc_model.pipeline.torch_sid = torch.tensor(
            [sid], device=vc_instance.vc_model.pipeline.device, dtype=torch.int64
        )

    # FAISS Index Tracking Path Search Block
    index_path = params.get("index_path", None)
    if index_path:
        if vc_instance.vc_model.index_path != index_path:
            from main.library.utils import load_faiss_index

            nprobe = params.get("nprobe", 1)
            # Sanitize paths to handle literal quote configurations securely
            index, index_reconstruct = load_faiss_index(
                index_path.strip().strip('"').strip("\n").strip('"').strip().replace("trained", "added"),
                nprobe=nprobe
            )

            vc_instance.vc_model.pipeline.index = index
            if vc_instance.vc_model.pipeline.index.index is not None: vc_instance.vc_model.pipeline.index.index.nprobe = nprobe
            vc_instance.vc_model.pipeline.big_tsr = index_reconstruct
            vc_instance.vc_model.index_path = index_path
    else:
        vc_instance.vc_model.pipeline.index = None
        vc_instance.vc_model.pipeline.big_tsr = None
        vc_instance.vc_model.index_path = None

    # Pitch Predictor and Feature Embedder Extraction Block
    f0_method = params.get("f0_method", vc_instance.vc_model.pipeline.f0_method)
    predictor_onnx = params.get("predictor_onnx", vc_instance.vc_model.pipeline.predictor.predictor_onnx)
    embedders = params.get("embedder_model", vc_instance.vc_model.embedder_model)
    embedders_mode = params.get("embedders_mode", vc_instance.vc_model.embedders_mode)
    custom_embedders = params.get("embedder_model_custom", None)
    embedder_model = (embedders if embedders != "custom" else custom_embedders)

    # Ensure files and parameters match required asset types
    from main.library.utils import check_assets
    check_assets(f0_method, embedders, predictor_onnx, embedders_mode)

    # Hot-swap the pitch predictor backend structures if options are updated
    if (
        vc_instance.vc_model.pipeline.vc.use_f0 and (
            vc_instance.vc_model.pipeline.f0_method != f0_method or
            vc_instance.vc_model.pipeline.predictor.predictor_onnx != predictor_onnx
        )
    ):
        old_predictor = vc_instance.vc_model.pipeline.predictor
        del old_predictor.pw, old_predictor.fcpe, old_predictor.djcm, old_predictor.penn, old_predictor.pesto, old_predictor.swift, old_predictor.rmvpe, old_predictor.crepe, old_predictor.mangio_penn, old_predictor.mangio_crepe
        del old_predictor

        vc_instance.vc_model.pipeline.predictor = vc_instance.vc_model.pipeline.setup_predictor(PIPELINE_SAMPLE_RATE, params.get("hop_length", vc_instance.vc_model.pipeline.predictor.hop_length), predictor_onnx)
        vc_instance.vc_model.pipeline.f0_method = f0_method
        vc_instance.vc_model.pipeline.predictor.predictor_onnx = predictor_onnx

    # Hot-swap Content Vec or Hubert feature extractors if modified
    if embedder_model:
        if (
            vc_instance.vc_model.embedder_model != embedder_model or
            vc_instance.vc_model.embedders_mode != embedders_mode
        ):
            old_embedder = vc_instance.vc_model.pipeline.embedder
            del old_embedder

            vc_instance.vc_model.pipeline.embedder = vc_instance.vc_model.pipeline.setup_embedder(embedder_model, embedders_mode)
            vc_instance.vc_model.embedder_model = embedder_model
            vc_instance.vc_model.embedders_mode = embedders_mode

    if (
        vc_instance.vc_model.pipeline.vc.architecture == "SVC" and
        vc_instance.vc_model.pipeline.vc.net_g.noise_scale != noise_scale
    ): 
        vc_instance.vc_model.pipeline.vc.net_g.noise_scale = noise_scale

@app.post("/record")
async def record(request: Request):
    """
    HTTP POST route acting as a remote trigger toggle to handle runtime sound recording sequences 
    of the converted audio streams.

    Args:
        request (Request): The incoming request payload containing parameters for target path and export format.

    Returns:
        dict: Operational response packet describing status modifications, info alerts, or warning messages.
    """

    global vc_instance

    data = await request.json()
    record_button = data.get("record_button", translations["stop_record"])
    record_audio_path = data.get("record_audio_path", None)
    export_format = data.get("export_format", "wav")

    if vc_instance is None:
        return {
            "type": "warnings",
            "value": translations["realtime_not_found"],
            "button": translations["start_record"],
            "path": None
        }

    # Evaluate execution parameters depending on toggle switch intent labels
    if record_button == translations["start_record"]:
        if not record_audio_path:
            record_audio_path = os.path.join(configs["audios_path"], "record_audio.wav")

        vc_instance.record_audio = True
        vc_instance.record_audio_path = record_audio_path
        vc_instance.export_format = export_format
        vc_instance.setup_soundfile_record()

        return {
            "type": "info",
            "value": translations["starting_record"],
            "button": translations["stop_record"],
            "path": None
        }
    else:
        vc_instance.record_audio = False
        vc_instance.record_audio_path = None
        vc_instance.soundfile = None

        return {
            "type": "info",
            "value": translations["stopping_record"],
            "button": translations["start_record"],
            "path": record_audio_path
        }

@app.websocket("/ws-audio")
async def websocket_audio(ws: WebSocket):
    """
    Main real-time audio pipeline hub handling binary frame ingestion, inference callback triggers, 
    latency logging metrics, and synthesized sample extraction.

    Args:
        ws (WebSocket): Continuous binary connection context loop.
    """

    global vc_instance, params
    await ws.accept()

    logger.info(translations["ws_connected"])

    try:
        text = await ws.receive_text()
        params = json.loads(text)

        read_chunk_size = int(params["chunk_size"])
        block_frame = read_chunk_size * 128
        embedders = params["embedders"]

        model_pth = params["model_pth"]
        model_pth = os.path.join(configs["weights_path"], model_pth) if not os.path.exists(model_pth) else model_pth

        # Structural validity check boundary to reject faulty setups
        if (
            not model_pth or 
            not os.path.exists(model_pth) or 
            os.path.isdir(model_pth) or 
            not model_pth.endswith((".pth", ".onnx"))
        ):
            logger.warning(translations["provide_model"])
            await ws.send_text(json.dumps({"type": "warnings", "value": translations["provide_model"]}))
            return
        
        logger.info(translations["start_realtime"])
        # Create the continuous VoiceChanger backend pipeline worker instance if not initialized
        if vc_instance is None:
            vc_instance = VoiceChanger(
                read_chunk_size=read_chunk_size, 
                cross_fade_overlap_size=params["cross_fade_overlap_size"], 
                input_sample_rate=DEVICE_SAMPLE_RATE, 
                output_sample_rate=DEVICE_SAMPLE_RATE, 
                extra_convert_size=params["extra_convert_size"],
                model_path=model_pth, 
                index_path=params["model_index"], 
                f0_method=params["f0_method"], 
                predictor_onnx=params["predictor_onnx"], 
                embedder_model=(embedders if embedders != "custom" else params["custom_embedders"]), 
                embedders_mode=params["embedders_mode"], 
                sample_rate=PIPELINE_SAMPLE_RATE, 
                hop_length=params["hop_length"], 
                silent_threshold=params["silent_threshold"],
                vad_enabled=params["vad_enabled"],
                vad_sensitivity=params["vad_sensitivity"],
                vad_frame_ms=params["vad_frame_ms"],
                clean_audio=params["clean_audio"], 
                clean_strength=params["clean_strength"],
                post_process=params["post_process"], 
                sid=params["sid"],
                noise_scale=params["noise_scale"],
                nprobe=params["nprobe"],
                record_audio=False,
                record_audio_path=None,
                export_format="wav",
                **params["kwargs"]
            )
        
        logger.info(translations["realtime_is_ready"])
        # Continuous listening loop tracking incoming raw binary floating-point stream blocks
        while 1:
            audio = await ws.receive_bytes()
            arr = np.frombuffer(audio, dtype=np.float32)

            # Enforce strict buffer alignment by padding or slicing anomalous incoming frames
            if arr.size != block_frame:
                arr = (
                    np.pad(arr, (0, block_frame - arr.size)).astype(np.float32) 
                    if arr.size < block_frame else 
                    arr[:block_frame].astype(np.float32)
                )

            if vc_instance is None: return

            # Apply gain amplification and forward the block into the pitch shifting & inference pipeline
            audio_output, vol, perf = vc_instance.on_request(
                arr * (params["input_audio_gain"] / 100.0), 
                f0_up_key=params["f0_up_key"], 
                index_rate=params["index_rate"], 
                protect=params["protect"], 
                filter_radius=params["filter_radius"], 
                rms_mix_rate=params["rms_mix_rate"], 
                f0_autotune=params["f0_autotune"], 
                f0_autotune_strength=params["f0_autotune_strength"], 
                proposal_pitch=params["proposal_pitch"], 
                proposal_pitch_threshold=params["proposal_pitch_threshold"],
                embedders_mix=params["embedders_mix"],
                embedders_mix_layers=params["embedders_mix_layers"],
                embedders_mix_ratio=params["embedders_mix_ratio"],
                use_phase_vocoder=params["use_phase_vocoder"]
            )

            # Asynchronously dispatch telemetry diagnostic parameters along with the processed binary chunk
            await ws.send_text(json.dumps({"type": "latency", "value": perf, "volume": vol}))
            await ws.send_bytes(audio_output.tobytes())
    except WebSocketDisconnect:
        logger.info(translations["ws_disconnected"])
    except Exception as e:
        import traceback

        logger.debug(traceback.format_exc())
        logger.info(translations["error_occurred"].format(e=e))
    finally:
        # Strict Memory Deallocation and GPU VRAM Cleanup Block
        if vc_instance is not None:
            del vc_instance.vc_model.pipeline, vc_instance.vc_model
            del vc_instance
            vc_instance = None

        # Force Python's garbage collector to release detached tensors
        gc.collect()
        clear_gpu_cache()

        # Safely shut down the socket handshake context loop
        try:
            await ws.close()
        except:
            pass

        logger.info(translations["ws_closed"])