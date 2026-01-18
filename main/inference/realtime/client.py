import os
import sys
import json

import numpy as np

from fastapi import FastAPI, WebSocketDisconnect, WebSocket, Request

sys.path.append(os.getcwd())

from main.library.utils import clear_gpu_cache
from main.app.variables import configs, translations, logger, config
from main.inference.realtime.realtime import VoiceChanger, RVC_Realtime

app = FastAPI()
vc_instance = None
params = {}

PIPELINE_SAMPLE_RATE = 16000
DEVICE_SAMPLE_RATE = 48000

@app.websocket("/change-config")
async def change_config(ws: WebSocket):
    global vc_instance, params

    if vc_instance is None: return
    await ws.accept()

    text = await ws.receive_text()
    jsons = json.loads(text)

    if jsons["if_kwargs"]:
        params["kwargs"][jsons["key"]] = jsons["value"]
    else:
        params[jsons["key"]] = jsons["value"]

    crossfade_frame = int(params.get("cross_fade_overlap_size", 0.1) * DEVICE_SAMPLE_RATE)
    extra_frame = int(params.get("extra_convert_size", 0.5) * DEVICE_SAMPLE_RATE)

    if (
        vc_instance.crossfade_frame != crossfade_frame or
        vc_instance.extra_frame != extra_frame
    ):
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

    vc_instance.vc_model.input_sensitivity = 10 ** (params.get("silent_threshold", -90) / 20)

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

    clean_audio = params.get("clean_audio", False)
    clean_strength = params.get("clean_strength", 0.5)

    if clean_audio is False:
        vc_instance.vc_model.tg = None
    elif clean_audio and vc_instance.vc_model.tg is None:
        from main.tools.noisereduce import TorchGate

        vc_instance.vc_model.tg = (
            TorchGate(
                vc_instance.vc_model.pipeline.tgt_sr,
                prop_decrease=clean_strength,
            ).to(config.device)
        )

    if vc_instance.vc_model.tg is not None:
        vc_instance.vc_model.tg.prop_decrease = clean_strength

    post_process = params.get("post_process", False)
    kwargs = params.get("kwargs", {})

    if post_process is False:
        vc_instance.vc_model.board = None
        vc_instance.vc_model.kwargs = None
    elif post_process and vc_instance.vc_model.kwargs != kwargs:
        new_board = vc_instance.vc_model.setup_pedalboard(**kwargs)
        vc_instance.vc_model.board = new_board
        vc_instance.vc_model.kwargs = kwargs.copy()

    model_pth = params.get("model_path", vc_instance.vc_model.model_path)
    model_pth = os.path.join(configs["weights_path"], model_pth) if not os.path.exists(model_pth) else model_pth

    if model_pth and vc_instance.vc_model.model_path != model_pth:
        vc_instance.vc_model.model_path = model_pth
        vc_instance.vc_model.pipeline.inference.get_synthesizer(model_pth)

        vc_instance.vc_model.pipeline.version = vc_instance.vc_model.pipeline.inference.version
        vc_instance.vc_model.pipeline.energy = vc_instance.vc_model.pipeline.inference.energy

        if vc_instance.vc_model.pipeline.inference.energy:
            from main.inference.extracting.rms import RMSEnergyExtractor

            rms = RMSEnergyExtractor(
                frame_length=2048, 
                hop_length=160, 
                center=True, 
                pad_mode="reflect"
            ).to(config.device).eval()

            vc_instance.vc_model.pipeline.rms = rms
        else:
            vc_instance.vc_model.pipeline.rms = None

    sid = params.get("sid", vc_instance.vc_model.pipeline.sid)
    if vc_instance.vc_model.pipeline.sid != sid:
        import torch
        vc_instance.vc_model.pipeline.torch_sid = torch.tensor(
            [sid], device=vc_instance.vc_model.pipeline.device, dtype=torch.int64
        )

    index_path = params.get("index_path", None)
    if index_path:
        if vc_instance.vc_model.index_path != index_path:
            from main.library.utils import load_faiss_index

            index, index_reconstruct = load_faiss_index(
                index_path.strip().strip('"').strip("\n").strip('"').strip().replace("trained", "added")
            )

            vc_instance.vc_model.pipeline.index = index
            vc_instance.vc_model.pipeline.big_npy = index_reconstruct
            vc_instance.vc_model.index_path = index_path
    else:
        vc_instance.vc_model.pipeline.index = None
        vc_instance.vc_model.pipeline.big_npy = None
        vc_instance.vc_model.index_path = None

    f0_method = params.get("f0_method", vc_instance.vc_model.pipeline.f0_method)
    predictor_onnx = params.get("predictor_onnx", vc_instance.vc_model.pipeline.predictor.predictor_onnx)
    embedders = params.get("embedder_model", vc_instance.vc_model.embedder_model)
    embedders_mode = params.get("embedders_mode", vc_instance.vc_model.embedders_mode)
    custom_embedders = params.get("embedder_model_custom", None)
    embedder_model = (embedders if embedders != "custom" else custom_embedders)

    from main.library.utils import check_assets
    check_assets(f0_method, embedders, predictor_onnx, embedders_mode)

    if (
        vc_instance.vc_model.pipeline.f0_method != f0_method or
        vc_instance.vc_model.pipeline.predictor.predictor_onnx != predictor_onnx
    ):
        old_predictor = vc_instance.vc_model.pipeline.predictor
        del old_predictor

        from main.library.predictors.Generator import Generator

        predictor = Generator(
            sample_rate=params.get("sample_rate", vc_instance.vc_model.sample_rate), 
            hop_length=params.get("f0_method", vc_instance.vc_model.pipeline.predictor.hop_length), 
            f0_min=50.0, 
            f0_max=1100.0, 
            alpha=0.5, 
            is_half=config.is_half, 
            device=config.device, 
            predictor_onnx=predictor_onnx, 
            delete_predictor_onnx=False
        )

        vc_instance.vc_model.pipeline.predictor = predictor
        vc_instance.vc_model.pipeline.f0_method = f0_method
        vc_instance.vc_model.pipeline.predictor.predictor_onnx = predictor_onnx

    if embedder_model:
        if (
            vc_instance.vc_model.embedder_model != embedder_model or
            vc_instance.vc_model.embedders_mode != embedders_mode
        ):
            old_embedder = vc_instance.vc_model.pipeline.embedder
            del old_embedder

            import torch
            from main.library.utils import load_embedders_model

            embedder = load_embedders_model(
                embedder_model, 
                embedders_mode=embedders_mode
            )

            if isinstance(embedder, torch.nn.Module): 
                dtype = torch.float16 if config.is_half else torch.float32
                embedder = embedder.to(config.device).to(dtype).eval()

            vc_instance.vc_model.pipeline.embedder = embedder
            vc_instance.vc_model.embedder_model = embedder_model
            vc_instance.vc_model.embedders_mode = embedders_mode

@app.post("/record")
async def record(request: Request):
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

    if record_button == translations["start_record"]:
        if not record_audio_path:
            record_audio_path = os.path.join("audios", "record_audio.wav")

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

        if (
            not model_pth or 
            not os.path.exists(model_pth) or 
            os.path.isdir(model_pth) or 
            not model_pth.endswith((".pth", ".onnx"))
        ):
            logger.warning(translations["provide_file"].format(filename=translations["model"]))
            await ws.send_text(json.dumps({"type": "warnings", "value": translations["provide_file"].format(filename=translations["model"])}))
            return
        
        logger.info(translations["start_realtime"])

        if vc_instance is None:
            vc_instance = VoiceChanger(
                read_chunk_size=read_chunk_size, 
                cross_fade_overlap_size=params["cross_fade_overlap_size"], 
                input_sample_rate=DEVICE_SAMPLE_RATE, 
                extra_convert_size=params["extra_convert_size"]
            )

            vc_instance.initialize(vc_model=RVC_Realtime(
                model_path=model_pth, 
                index_path=params["model_index"], 
                f0_method=params["f0_method"], 
                predictor_onnx=params["predictor_onnx"], 
                embedder_model=(embedders if embedders != "custom" else params["custom_embedders"]), 
                embedders_mode=params["embedders_mode"], 
                sample_rate=PIPELINE_SAMPLE_RATE, 
                hop_length=params["hop_length"], 
                silent_threshold=params["silent_threshold"], 
                input_sample_rate=DEVICE_SAMPLE_RATE, 
                output_sample_rate=DEVICE_SAMPLE_RATE, 
                vad_enabled=params["vad_enabled"], 
                vad_sensitivity=params["vad_sensitivity"], 
                vad_frame_ms=params["vad_frame_ms"], 
                clean_audio=params["clean_audio"], 
                clean_strength=params["clean_strength"],
                post_process=params["post_process"],
                sid=params["sid"],
                **params["kwargs"]
            ))
        
        logger.info(translations["realtime_is_ready"])

        while 1:
            audio = await ws.receive_bytes()
            arr = np.frombuffer(audio, dtype=np.float32)

            if arr.size != block_frame:
                arr = (
                    np.pad(arr, (0, block_frame - arr.size)).astype(np.float32) 
                    if arr.size < block_frame else 
                    arr[:block_frame].astype(np.float32)
                )

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
                proposal_pitch_threshold=params["proposal_pitch_threshold"]
            )

            await ws.send_text(json.dumps({"type": "latency", "value": perf[1], "volume": vol}))
            await ws.send_bytes(audio_output.tobytes())
    except WebSocketDisconnect:
        logger.info(translations["ws_disconnected"])
    except Exception as e:
        import traceback

        logger.debug(traceback.format_exc())
        logger.info(translations["error_occurred"].format(e=e))
    finally:
        if vc_instance is not None:
            del vc_instance
            vc_instance = None

        clear_gpu_cache()

        try:
            await ws.close()
        except:
            pass

        logger.info(translations["ws_closed"])