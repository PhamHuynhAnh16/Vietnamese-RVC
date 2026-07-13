import os
import re
import sys
import json
import torch
import shutil

import gradio as gr

sys.path.append(os.getcwd())

import main.library.audio.sounddevice as sd

from main.inference.realtime.audio import audio_device

from main.app.variables import (
    logger, 
    config, 
    configs, 
    edgetts, 
    method_f0, 
    vr_models, 
    mdx_models, 
    file_types, 
    configs_json, 
    translations, 
    demucs_models, 
    method_f0_full, 
    gradio_version, 
    google_tts_voice 
)

# Initialize GPU and audio device maps from configuration settings
ngpu, gpu_infos = config.get_gpu_list()
input_channels_map, output_channels_map = audio_device()

def gr_info(message=""):
    """Displays an information message in the Gradio UI and logs it."""

    gr.Info(message, duration=2)
    logger.info(message)

def gr_warning(message=""):
    """Displays a warning message in the Gradio UI and logs it."""

    gr.Warning(message, duration=2)
    logger.warning(message)

def gr_error(message=""):
    """Displays an error message in the Gradio UI and logs it."""

    gr.Error(message=message, duration=6)
    logger.error(message)

def get_gpu_info():
    """Retrieves formatted GPU info string and total GPU count based on system status."""

    return (("\n".join(gpu_infos) if len(gpu_infos) > 0 and not config.cpu_mode else translations["no_support_gpu"]), ngpu)

def gpu_number_str():
    """Generates a string representation of available GPU indices or fallback states."""

    if config.cpu_mode: return "-"
    return str("-".join(map(str, range(ngpu)))) if config.device.startswith("cuda") else str(config.pytorch_device_idx)

def change_f0_choices(): 
    """Scans the designated f0 directory to update available text choices in the UI."""

    # Find all absolute paths of .txt files recursively inside the F0 path
    f0_file = sorted([
        os.path.abspath(os.path.join(root, f)) 
        for root, _, files in os.walk(configs["f0_path"]) 
        for f in files 
        if f.endswith(".txt")
    ])

    return {
        "value": f0_file[0] if len(f0_file) >= 1 else "", 
        "choices": f0_file, 
        "__type__": "update"
    }

def change_audios_choices(input_audio=""): 
    """Scans the audios directory to dynamically populate audio file choices in the UI."""

    # Fetch all files matching supported extension formats excluding output tags
    audios = sorted([
        os.path.abspath(os.path.join(root, f)) 
        for root, _, files in os.walk(configs["audios_path"]) 
        for f in files if os.path.splitext(f)[1].lower() in file_types and not f.startswith(("Convert", "output"))
    ])

    return {
        "value": input_audio if input_audio != "" else (audios[0] if len(audios) >= 1 else ""), 
        "choices": audios, 
        "__type__": "update"
    }

def change_reference_choices():
    """Scans the reference directory and clean up folder names using regex patterns."""

    # Filter directories and remove suffix tracking tags like _v1_hubert_base_True 
    reference = sorted([
        re.sub(r'_v\d+_(?:[A-Za-z0-9_]+?)_(True|False)$', '', name) 
        for name in os.listdir(configs["reference_path"]) 
        if (
            os.path.exists(os.path.join(configs["reference_path"], name)) and 
            os.path.isdir(os.path.join(configs["reference_path"], name))
        )
    ])

    return {
        "value": reference[0] if len(reference) >= 1 else "", 
        "choices": reference, 
        "__type__": "update"
    }

def change_models_choices():
    """Gathers available weight files (.pth/.onnx) and index files from the logs folder."""

    model, index = (
        sorted([
            model 
            for model in os.listdir(configs["weights_path"]) 
            if model.endswith((".pth", ".onnx")) and not model.startswith("G_") and not model.startswith("D_")
        ]), 
        sorted([
            os.path.join(root, name) 
            for root, _, files in os.walk(configs["logs_path"], topdown=False) 
            for name in files if name.endswith(".index") and "trained" not in name
        ])
    )

    return [
        {
            "value": model[0] if len(model) >= 1 else "", 
            "choices": model, 
            "__type__": "update"
        }, 
        {
            "value": index[0] if len(index) >= 1 else "", 
            "choices": index, 
            "__type__": "update"
        }
    ]

def change_pretrained_choices():
    """Separates custom pretrained models into discriminator (D) and generator (G) sets."""

    pretrainD = sorted([
        model for model in os.listdir(configs["pretrained_custom_path"]) 
        if model.endswith(".pth") and "D" in model
    ])

    pretrainG = sorted([
        model for model in os.listdir(configs["pretrained_custom_path"]) 
        if model.endswith(".pth") and "G" in model
    ])

    return [
        {
            "choices": pretrainD, 
            "value": pretrainD[0] if len(pretrainD) >= 1 else "", 
            "__type__": "update"
        }, 
        {
            "choices": pretrainG, 
            "value": pretrainG[0] if len(pretrainG) >= 1 else "", 
            "__type__": "update"
        }
    ]

def change_preset_choices():
    """Fetches preset configuration files dedicated to file conversion."""

    return {
        "value": "", 
        "choices": sorted([
            f for f in os.listdir(configs["presets_path"]) 
            if f.endswith(".conversion.json")
        ]), 
        "__type__": "update"
    }

def change_effect_preset_choices():
    """Fetches preset configuration files dedicated to post-processing effects."""

    return {
        "value": "", 
        "choices": sorted([
            f for f in os.listdir(configs["presets_path"]) 
            if f.endswith(".effect.json")
        ]), 
        "__type__": "update"
    }

def change_realtime_preset_choices():
    """Fetches preset configuration files dedicated to realtime execution."""

    return {
        "value": "", 
        "choices": sorted([
            f for f in os.listdir(configs["presets_path"]) 
            if f.endswith(".realtime.json")
        ]), 
        "__type__": "update"
    }

def change_tts_voice_choices(google=False):
    """Updates Voice choices depending on whether Google TTS or Edge TTS is chosen."""

    return {
        "choices": google_tts_voice if google else edgetts, 
        "value": google_tts_voice[0] if google else edgetts[0], 
        "__type__": "update"
    }

def change_backing_choices(
    backing=False, 
    not_merge=False
):
    """Controls interactive permissions of sub-elements based on merge and backing configurations."""

    if backing or not_merge: 
        return {
            "value": False, 
            "interactive": False, 
            "__type__": "update"
        }
    elif not backing or not not_merge: 
        return  {
            "interactive": True, 
            "__type__": "update"
        }
    else: 
        gr_warning(translations["option_not_valid"])

def change_download_choices(select = translations["download_url"]):
    """Toggles visible interface boxes inside the main downloading tab depending on the source selected."""

    selects = [False]*10

    # Index mapping definitions for visibility flags
    if select == translations["download_url"]: 
        selects[0] = selects[1] = selects[2] = True
    elif select == translations["download_from_csv"]:  
        selects[3] = selects[4] = True
    elif select == translations["search_models"]: 
        selects[5] = selects[6] = True
    elif select == translations["upload"]: 
        selects[9] = True
    else: 
        gr_warning(translations["option_not_valid"])

    return [
        {
            "visible": selects[i] if gradio_version else (selects[i] or "hidden"), 
            "__type__": "update"
        } 
        for i in range(len(selects))
    ]

def change_download_pretrained_choices(select = translations["download_url"]):
    """Toggles visibility parameters inside the pretrained model downloading panel section."""

    selects = [False]*7

    if select == translations["download_url"]: 
        selects[0] = selects[1] = selects[2] = True
    elif select == translations["list_model"]: 
        selects[3] = selects[4] = selects[5] = True
    elif select == translations["upload"]: 
        selects[6] = True
    else: 
        gr_warning(translations["option_not_valid"])

    return [
        {
            "visible": selects[i] if gradio_version else (selects[i] or "hidden"), 
            "__type__": "update"
        } 
        for i in range(len(selects))
    ]

def get_index(model = ""):
    """Auto-detects and matches the correct retrieval index file path belonging to a chosen model name."""

    model = os.path.basename(model).split("_")[0]

    return {
        "value": next((
            f for f in [
                os.path.join(root, name) 
                for root, _, files in os.walk(configs["logs_path"], topdown=False) 
                for name in files 
                if name.endswith(".index") and "trained" not in name
            ] 
            if model.split(".")[0] in f
        ), ""), 
        "__type__": "update"
    } if model else None

def index_strength_show(index = ""):
    """Reveals or masks Index processing intensity adjustments if a valid file target is allocated."""

    index_strength_visible = (
        index != "" and 
        index != None and 
        os.path.exists(index) and 
        os.path.isfile(index)
    )

    return [
        {
            "visible": index_strength_visible if gradio_version else (index_strength_visible or "hidden"), 
            "value": 0.5, 
            "__type__": "update"
        },
        {
            "visible": index_strength_visible if gradio_version else (index_strength_visible or "hidden"), 
            "value": 9, 
            "__type__": "update"
        }
    ]

def hoplength_show(
    method="rmvpe", 
    hybrid_method=None
):
    """Determines whether the Hop Length parameter control needs to be visible based on pitch tracking methods."""

    visible = False
    # Hop length adjustment is necessary only for specific pitch estimation backends
    for m in [
        "mangio-crepe", 
        "yin", 
        "piptrack", 
        "mangio-penn"
    ]:
        if m in method: visible = True

        if (
            hybrid_method is not None and 
            m in hybrid_method
        ): 
            visible = True

        if visible: break
        else: visible = False
    
    return {
        "visible": visible if gradio_version else (visible or "hidden"), 
        "__type__": "update"
    }

def visible(value=False):
    """Generates direct layout display flag mappings based on component status."""

    return {
        "visible": value if gradio_version else (value or "hidden"), 
        "__type__": "update"
    }

def visibleFalse(value=False):
    """Sets component visibility property alongside a forced initialization value of False."""

    return {
        "visible": value if gradio_version else (value or "hidden"), 
        "value": False, 
        "__type__": "update"
    }

def valueFalse_interactive(value=False): 
    """Enforces active component interactivity permissions accompanied with a cleared selection flag."""

    return {
        "value": False, 
        "interactive": value, 
        "__type__": "update"
    }

def interactive(value=False): 
    """Sets a component's interactive status."""

    return {
        "interactive": value, 
        "__type__": "update"
    }

def valueEmpty_visible1(value=False): 
    """Clears value contents to empty text strings while managing overall item display parameters."""

    return {
        "value": "", 
        "visible": value if gradio_version else (value or "hidden"), 
        "__type__": "update"
    }

def pitch_guidance_lock(vocoders = "Default"):
    """Locks pitch guidance to True and restricts interactive changes if using non-default vocoders."""

    return {
        "value": True, 
        "interactive": vocoders == "Default", 
        "__type__": "update"
    }

def vocoders_lock(pitch=True):
    """Blocks vocoder optimization configuration fields if Pitch calculation parameters are omitted."""

    if pitch:
        return gr.update(interactive=True)

    return gr.update(
        value="Default",
        interactive=False
    )

def unlock_f0(value=False):
    """Dynamically fills pitch processing selection choices based on expanded algorithm capabilities."""

    return {
        "choices": method_f0_full if value else method_f0, 
        "value": "rmvpe", 
        "__type__": "update"
    } 

def change_fp(fp="fp32", bf16=False, tf32=False, int8=False):
    """Validates hardware support and updates runtime compute tensor precision parameters in configs_json."""

    global config

    fp16 = fp == "fp16"
    # Abort tracking state adjustments if target architecture does not support FP16 operations
    if fp16 and not config.allow_is_half: 
        gr_warning(translations["fp16_not_support"])
        return "fp32"
    else:
        gr_info(translations["start_update_precision"])
        # Safely overwrite local states and dump serialized data structures back to file
        configs = json.load(open(configs_json, "r"))
        configs["fp16"] = config.is_half = fp16
        configs["brain"] = config.brain = bf16
        configs["tf32"] = config.tf32 = tf32
        configs["int8"] = config.int8 = int8

        with open(configs_json, "w") as f:
            json.dump(configs, f, indent=4)

        gr_info(translations["success"])
        return "fp16" if fp16 else "fp32"
    
def process_output(file_path = ""):
    """Manages output tracking filenames by clearing existing items or appending indexing strings."""

    if config.configs.get("delete_exists_file", True):
        # Delete existing target files directly if overwriting is permitted
        if os.path.exists(file_path) and os.path.isfile(file_path): os.remove(file_path)
        return file_path
    else:
        # Append incremented indexing values sequentially until finding a safe path destination location
        if not os.path.exists(file_path): return file_path
        file = os.path.splitext(os.path.basename(file_path))

        index = 1
        while 1:
            file_path = os.path.join(
                os.path.dirname(file_path), 
                f"{file[0]}_{index}{file[1]}"
            )

            if not os.path.exists(file_path): return file_path
            index += 1

def shutil_move(input_path = "", output_path = ""):
    """Safely relocates files across local directories using atomic collision checking hooks."""

    output_path = (
        os.path.join(output_path, os.path.basename(input_path)) 
        if os.path.isdir(output_path) else 
        output_path
    )

    return (
        shutil.move(input_path, process_output(output_path)) 
        if os.path.exists(output_path) else 
        shutil.move(input_path, output_path)
    )

def separate_change(
    model_name = "HT-Tuned", 
    karaoke_model = "MDX-Version-2", 
    reverb_model = "MDX-Reverb", 
    enable_post_process = False, 
    separate_backing = False, 
    separate_reverb = False, 
    enable_denoise = False
):
    """Evaluates user-selected stem separation model classes to dynamically configure visible UI sliders."""

    # Detect algorithmic source framework variants
    model_type = (
        "vr" if model_name in list(vr_models.keys()) else "mdx" 
        if model_name in list(mdx_models.keys()) else "demucs" 
        if model_name in list(demucs_models.keys()) else ""
    )

    karaoke_type = ("vr" if karaoke_model.startswith("VR") else "mdx") if separate_backing else None
    reverb_type = ("vr" if not reverb_model.startswith("MDX") else "mdx") if separate_reverb else None

    all_types = {model_type, karaoke_type, reverb_type}

    is_vr = "vr" in all_types
    is_mdx = "mdx" in all_types
    is_demucs = "demucs" in all_types

    # Returns an array mapped to expected Gradio component tracking behaviors sequentially
    return [
        visible(separate_backing),
        visible(separate_reverb),
        visible(is_mdx or is_demucs),
        visible(is_mdx or is_demucs),
        visible(is_mdx),
        visible(is_mdx or is_vr),
        visible(is_demucs),
        visible(is_vr),
        visible(is_vr),
        visible(is_vr and enable_post_process),
        visible(is_vr and enable_denoise),
        valueFalse_interactive(is_vr),
        valueFalse_interactive(is_vr),
        interactive(is_vr) if is_vr else valueFalse_interactive(is_vr)
    ]

def create_dataset_change(
    model_name = "HT-Tuned", 
    reverb_model = "MDX-Reverb", 
    enable_post_process = False, 
    separate_reverb = False, 
    enable_denoise = False
):
    """Adapts dataset processing workflow variables dynamically based on running separation backends."""

    model_type = (
        "vr" if model_name in list(vr_models.keys()) else "mdx" 
        if model_name in list(mdx_models.keys()) else "demucs" 
        if model_name in list(demucs_models.keys()) else ""
    )

    reverb_type = ("vr" if not reverb_model.startswith("MDX") else "mdx") if separate_reverb else None
    all_types = {model_type, reverb_type}

    is_vr = "vr" in all_types
    is_mdx = "mdx" in all_types
    is_demucs = "demucs" in all_types

    return [
        visible(separate_reverb),
        visible(is_mdx or is_demucs),
        visible(is_mdx or is_demucs),
        visible(is_mdx),
        visible(is_mdx or is_vr),
        visible(is_demucs),
        visible(is_vr),
        visible(is_vr),
        visible(is_vr and enable_post_process),
        visible(is_vr and enable_denoise),
        valueFalse_interactive(is_vr),
        valueFalse_interactive(is_vr),
        interactive(is_vr) if is_vr else valueFalse_interactive(is_vr)
    ]

def update_audio_device(input_device = "", output_device = "", monitor_device = "", monitor = False):
    """Initializes ASIO device channels layout mappings or falls back to traditional stereo sound rules."""

    input_is_asio = "ASIO" in input_device if input_device else (False if gradio_version else "hidden")
    output_is_asio = "ASIO" in output_device if output_device else (False if gradio_version else "hidden")
    monitor_is_asio = "ASIO" in monitor_device if monitor_device else (False if gradio_version else "hidden")

    # Reset structural sounddevice core drivers properties safely if ASIO is discovered
    if input_is_asio or output_is_asio or monitor_is_asio:
        sd.terminate()
        sd.initialize()

        try:
            input_max_ch = input_channels_map.get(input_device, [])[1]
            output_max_ch = output_channels_map.get(output_device, [])[1]
            monitor_max_ch = output_channels_map.get(monitor_device, [])[1] if monitor else 128
        except:
            input_max_ch = output_max_ch = monitor_max_ch = 2
    else: input_max_ch = output_max_ch = monitor_max_ch = 2

    return [
        visible(monitor),
        visible(monitor),
        visible(
            input_is_asio or output_is_asio or monitor_is_asio
        ),
        gr.update(
            visible=input_is_asio, 
            maximum=input_max_ch - 1
        ),
        gr.update(
            visible=output_is_asio,
            maximum=output_max_ch - 1
        ),
        gr.update(
            visible=monitor_is_asio, 
            maximum=monitor_max_ch - 1
        ),
        gr.update(
            value=resolve_sample_rate(input_channels_map[input_device][0]) if input_is_asio else 48000
        ),
        gr.update(
            value=resolve_sample_rate(output_channels_map[output_device][0]) if output_is_asio else 48000
        ),
        gr.update(
            visible=monitor, 
            value=resolve_sample_rate(output_channels_map[monitor_device][0]) if monitor_is_asio else 48000
        ),
        gr.update(
            visible=output_is_asio, 
            value=True
        ),
        gr.update(
            visible=monitor_is_asio, 
            value=True
        ),
    ]

def change_audio_device_choices():
    """Rescans host audio drivers to update input/output device options in Gradio selectors."""

    global input_channels_map, output_channels_map

    # Safely cycle operational status loops inside audio subsystems
    sd.terminate()
    sd.initialize()

    input_channels_map, output_channels_map = audio_device()
    input_channels_map_list, output_channels_map_list = list(input_channels_map.keys()), list(output_channels_map.keys())

    return [
        {
            "value": input_channels_map_list[0] if len(input_channels_map_list) >= 1 else "", 
            "choices": input_channels_map_list, 
            "__type__": "update"
        }, 
        {
            "value": output_channels_map_list[0] if len(output_channels_map_list) >= 1 else "", 
            "choices": output_channels_map_list, 
            "__type__": "update"
        },
        {
            "value": output_channels_map_list[0] if len(output_channels_map_list) >= 1 else "", 
            "choices": output_channels_map_list, 
            "__type__": "update"
        }
    ]

def resolve_sample_rate(input_device_id = 0):
    """Queries hardware drivers to extract default internal operational sample rates values."""

    if configs.get("asio_enabled", False): return 48000

    try:
        device = sd.query_devices(input_device_id)
        return int(device["default_samplerate"])
    except Exception:
        return 48000

def replace_punctuation(filename = ""):
    """Sanitizes file titles by purging illegal operational symbols with safe underscore placeholders."""

    return re.sub(r"_+", "_", re.sub(r"[ \-|]+", "_", filename.translate(str.maketrans("", "", '()[],"\'{}<>?*:/\\')))).strip("_")

def replace_url(url = ""):
    """Formats huggingface storage pathways links into direct execution links formats."""

    return url.replace("/blob/", "/resolve/").replace("?download=true", "").strip()

def replace_modelname(modelname = ""):
    """Strips extension suffixes from common AI models to reveal pure speaker tracking titles."""

    return replace_punctuation(
        modelname.replace(".onnx", "").replace(".pth", "").replace(".index", "").replace(".zip", "")
    )

def replace_export_format(audio_path = "", export_format = "wav"):
    """Alters structural configuration extensions targets mapped to specific encoding files parameters."""

    export_format = f".{export_format}"

    return (
        audio_path 
        if audio_path.endswith(export_format) else 
        audio_path.replace(f".{os.path.basename(audio_path).split('.')[-1]}", export_format)
    )

def update_dropdowns_from_json(data = {}):
    """Parses JSON data received from frontend JavaScript to dynamically update the choices and values of three audio device dropdowns."""
    if not data:
        return [
            gr.update(choices=[], value=None), 
            gr.update(choices=[], value=None), 
            gr.update(choices=[], value=None)
        ]

    inputs = list(data.get("inputs", {}).keys())
    outputs = list(data.get("outputs", {}).keys())

    return [
        gr.update(
            choices=inputs, 
            value=inputs[0] if len(inputs) > 0 else None
        ),
        gr.update(
            choices=outputs, 
            value=outputs[0] if len(outputs) > 0 else None
        ),
        gr.update(
            choices=outputs, 
            value=outputs[0] if len(outputs) > 0 else None
        ),
    ]

def update_button_from_json(data = {}):
    """Updates the interactivity states (enabled/disabled) of workflow buttons based on the control state flags passed from the frontend JavaScript."""
    if not data:
        return [
            gr.update(interactive=True), 
            gr.update(interactive=False)
        ]
    
    return [
        gr.update(interactive=data.get("start_button", True)),
        gr.update(interactive=data.get("stop_button", False))
    ]

def update_value_from_json(data = {}):
    """Updates UI text labels and destination paths dynamically using the tracking information generated by the frontend JavaScript execution."""

    if not data:
        return [
            gr.update(), 
            gr.update(value=None)
        ]
    
    return [
        data.get("button", "Start"),
        data.get("path", None),
    ]

def get_speakers_id_and_architecture(model = ""):
    """Reads structural internal tensors data parameters fields inside saved model weights to parse capabilities."""

    model_path = os.path.join(configs["weights_path"], model) if not os.path.exists(model) else model
    # Validate file integrity conditions before performing serialization inspections steps
    if not model or not os.path.exists(model_path) or os.path.isdir(model_path) or not model.endswith((".pth", ".onnx")): 
        return [
            {
                "visible": False if gradio_version else "hidden", 
                "value": 0, 
                "choices": [0], 
                "__type__": "update"
            },
            {
                "visible": False if gradio_version else "hidden",
                "value": 0.4,
                "__type__": "update"
            }
        ]

    try:
        if model_path.endswith(".pth"):
            # Inspect metadata fields from PyTorch checkpoints safely using cpu isolation configurations
            model_data = torch.load(model_path, map_location="cpu", weights_only=True)
        else:
            import onnx

            model_data = None
            # Scan structural attributes lists within ONNX metadata wrappers pipelines structures
            model = onnx.load(model_path)

            for prop in model.metadata_props:
                if prop.key == "model_info":
                    model_data = json.loads(prop.value)
                    break
            
            del model

        speakers_id = model_data.get("speakers_id", 1)
        speakers_id_visible = speakers_id and speakers_id != 1
        noise_scale_visible = model_data.get("architecture", "RVC") == "SVC"
        del model_data

        return [
            {
                "visible": speakers_id_visible if gradio_version else (speakers_id_visible or "hidden"), 
                "value": 0, 
                "choices": list(range(speakers_id)),
                "__type__": "update"
            },
            {
                "visible": noise_scale_visible if gradio_version else (noise_scale_visible or "hidden"),
                "value": 0.4,
                "__type__": "update"
            }
        ]
    except Exception:
        # Gracefully fall back to standard RVC configuration properties if internal validation parsing errors occur
        return [
            {
                "visible": False if gradio_version else "hidden", 
                "value": 0, 
                "choices": [0], 
                "__type__": "update"
            },
            {
                "visible": False if gradio_version else "hidden",
                "value": 0.4,
                "__type__": "update"
            }
        ]