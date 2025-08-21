import os
import sys
import time
import torch

sys.path.append(os.getcwd())

from main.library.utils import clear_gpu_cache
from main.app.core.ui import gr_info, gr_warning
from main.app.variables import translations, configs
from main.inference.realtime.callbacks import AudioCallbacks

running, callbacks, audio_manager = False, None, None

PIPELINE_SAMPLE_RATE = 16000
DEVICE_SAMPLE_RATE = 48000

interactive_true = {"interactive": True, "__type__": "update"}
interactive_false = {"interactive": False, "__type__": "update"}

def realtime_start(
    monitor,
    exclusive_mode,
    vad_enabled,
    input_audio_device,
    output_audio_device,
    monitor_output_device,
    input_audio_gan,
    output_audio_gan,
    monitor_audio_gan,
    input_asio_channels,
    output_asio_channels,
    monitor_asio_channels,
    chunk_size,
    pitch,
    model_pth,
    model_index,
    index_strength,
    onnx_f0_mode,
    f0_method,
    hop_length,
    embed_mode,
    embedders,
    custom_embedders,
    f0_autotune,
    proposal_pitch,
    f0_autotune_strength,
    proposal_pitch_threshold,
    rms_mix_rate,
    protect,
    filter_radius,
    silent_threshold,
    extra_convert_size,
    cross_fade_overlap_size,
    vad_sensitivity,
    vad_frame_ms
):
    global running, callbacks, audio_manager
    running = True

    gr_info(translations["start_realtime"])
    yield translations["start_realtime"], interactive_false, interactive_true

    if not input_audio_device or not output_audio_device:
        gr_warning(translations["provide_audio_device"])
        return translations["provide_audio_device"], interactive_true, interactive_false

    if monitor and not monitor_output_device:
        gr_warning(translations["provide_monitor_device"])
        return translations["provide_monitor_device"], interactive_true, interactive_false

    model_pth = os.path.join(configs["weights_path"], model_pth) if not os.path.exists(model_pth) else model_pth
    embedder_model = (embedders if embedders != "custom" else custom_embedders)

    if not model_pth or not os.path.exists(model_pth) or os.path.isdir(model_pth) or not model_pth.endswith((".pth", ".onnx")):
        gr_warning(translations["provide_file"].format(filename=translations["model"]))
        return translations["provide_file"].format(filename=translations["model"]), interactive_true, interactive_false

    input_audio_gan /= 10
    input_device_id = int(input_audio_device.split(":")[0])

    output_audio_gan /= 10
    output_device_id = int(output_audio_device.split(":")[0])

    monitor_audio_gan /= 10
    output_monitor_id = int(monitor_output_device.split(":")[0]) if monitor else None

    chunk_size = chunk_size * DEVICE_SAMPLE_RATE / 1000 / 128

    callbacks = AudioCallbacks(
        pass_through=False, 
        read_chunk_size=chunk_size, 
        cross_fade_overlap_size=cross_fade_overlap_size, 
        input_sample_rate=DEVICE_SAMPLE_RATE, 
        output_sample_rate=DEVICE_SAMPLE_RATE, 
        extra_convert_size=extra_convert_size, 
        model_path=model_pth, 
        index_path=model_index, 
        f0_method=f0_method, 
        f0_onnx=onnx_f0_mode, 
        embedder_model=embedder_model, 
        embedders_mode=embed_mode, 
        sample_rate=PIPELINE_SAMPLE_RATE, 
        hop_length=hop_length, 
        silent_threshold=silent_threshold, 
        f0_up_key=pitch, 
        index_rate=index_strength, 
        protect=protect, 
        filter_radius=filter_radius, 
        rms_mix_rate=rms_mix_rate,
        f0_autotune=f0_autotune, 
        f0_autotune_strength=f0_autotune_strength, 
        proposal_pitch=proposal_pitch, 
        proposal_pitch_threshold=proposal_pitch_threshold,
        input_audio_gan=input_audio_gan, 
        output_audio_gan=output_audio_gan,
        monitor_audio_gan=monitor_audio_gan,
        monitor=monitor,
        vad_enabled=vad_enabled,
        vad_sensitivity=vad_sensitivity,
        vad_frame_ms=vad_frame_ms
    )

    audio_manager = callbacks.audio
    audio_manager.start(
        input_device_id=input_device_id, 
        output_device_id=output_device_id, 
        output_monitor_id=output_monitor_id, 
        exclusive_mode=exclusive_mode, 
        asio_input_channel=input_asio_channels, 
        asio_output_channel=output_asio_channels, 
        asio_output_monitor_channel=monitor_asio_channels,
        read_chunk_size=chunk_size, 
        input_audio_sample_rate=DEVICE_SAMPLE_RATE, 
        output_monitor_sample_rate=DEVICE_SAMPLE_RATE
    )

    gr_info(translations["realtime_is_ready"])

    while running and callbacks is not None and audio_manager is not None:
        time.sleep(0.1)
        if hasattr(callbacks, "latency"): yield f"{translations['latency']}: {callbacks.latency:.2f} ms", interactive_false, interactive_true

    return translations["realtime_has_stop"], interactive_true, interactive_false

def realtime_stop():
    global running, callbacks, audio_manager

    if running and audio_manager is not None and callbacks is not None:
        gr_info(translations["stop_realtime"])

        audio_manager.stop()
        running = False

        if hasattr(callbacks, "latency"): del callbacks.latency
        del audio_manager, callbacks

        audio_manager = callbacks = None
        gr_info(translations["realtime_has_stop"])

        with torch.no_grad():
            clear_gpu_cache()

        return translations["realtime_has_stop"], interactive_true, interactive_false
    else:
        gr_warning(translations["realtime_not_found"])

        return translations["realtime_not_found"], interactive_true, interactive_false