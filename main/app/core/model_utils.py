import os
import sys
import json
import torch
import datetime

sys.path.append(os.getcwd())

from main.app.core.ui import gr_info, gr_warning, gr_error
from main.app.variables import config, logger, translations, configs

def fushion_model_pth(
    name, 
    model_path_1, 
    model_path_2, 
    ratio
):
    """
    Fuses two RVC (.pth) models together based on a specified blending ratio.

    This function extracts the weights from both checkpoints, validates their architectures,
    interpolates the weights linearly, and saves the new blended model.

    Args:
        name (str): The filename for the output fused model.
        model_path_1 (str): Path or filename of the first model.
        model_path_2 (str): Path or filename of the second model.
        ratio (float): The blending weight for the first model (usually between 0.0 and 1.0).

    Returns:
        list: A list containing a status message (str) and the output file path (str or None).
    """

    # Enforce the correct file extension for the output model
    if not name.endswith(".pth"): name = name + ".pth"
    # Resolve absolute paths using the configured weights directory if paths do not exist directly
    model_path_1 = os.path.join(configs["weights_path"], model_path_1) if not os.path.exists(model_path_1) else model_path_1
    model_path_2 = os.path.join(configs["weights_path"], model_path_2) if not os.path.exists(model_path_2) else model_path_2

    # Validate existence and format of the first model path
    if (
        not model_path_1 or 
        not os.path.exists(model_path_1) or 
        not model_path_1.endswith(".pth")
    ):
        gr_warning(translations["provide_model_idx"].format(idx="1"))
        return [translations["provide_model_idx"].format(idx="1"), None]
    
    # Validate existence and format of the second model path
    if (
        not model_path_2 or 
        not os.path.exists(model_path_2) or 
        not model_path_2.endswith(".pth")
    ):
        gr_warning(translations["provide_model_idx"].format(idx="2"))
        return [translations["provide_model_idx"].format(idx="2"), None]

    from collections import OrderedDict

    def extract(ckpt):
        """Helper function to extract clean model weights, ignoring recognition/encoder keys."""

        a = ckpt["model"]
        opt = OrderedDict()
        opt["weight"] = {}

        for key in a.keys():
            # Skip encoder query keys to avoid redundant configurations during fusion
            if "enc_q" in key: continue

            opt["weight"][key] = a[key]

        return opt
    
    try:
        # Load checkpoints onto CPU safely to avoid out-of-memory errors
        ckpt1 = torch.load(model_path_1, map_location="cpu", weights_only=True)
        ckpt2 = torch.load(model_path_2, map_location="cpu", weights_only=True)

        # Ensure both models share the same sample rate (sr) before fusion
        if ckpt1["sr"] != ckpt2["sr"]: 
            del ckpt1, ckpt2

            gr_warning(translations["sr_not_same"])
            return [translations["sr_not_same"], None]

        # Extract structural metadata from the first checkpoint
        cfg = ckpt1["config"]
        cfg_f0 = ckpt1["f0"]
        cfg_version = ckpt1["version"]
        cfg_sr = ckpt1["sr"]

        vocoder = ckpt1.get("vocoder", "Default")
        # Normalize the internal structure to extract raw weights dictionary
        ckpt1 = extract(ckpt1) if "model" in ckpt1 else ckpt1["weight"]
        ckpt2 = extract(ckpt2) if "model" in ckpt2 else ckpt2["weight"]

        # Structural sanity check: ensure both models contain the exact same layers
        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())): 
            gr_warning(translations["architectures_not_same"])

            return [translations["architectures_not_same"], None]
         
        gr_info(translations["start_fushion_model"])

        opt = OrderedDict()
        opt["weight"] = {}
        # Loop through all layers and perform linear interpolation (weighted average)
        for key in ckpt1.keys():
            # Special handling for embedding layer if speaker counts or configurations differ
            if key == "emb_g.weight" and ckpt1[key].shape != ckpt2[key].shape:
                min_shape0 = min(ckpt1[key].shape[0], ckpt2[key].shape[0])
                # Fuse up to the minimum matching dimensions
                opt["weight"][key] = (
                    ratio * (ckpt1[key][:min_shape0].float()) + (1 - ratio) * (ckpt2[key][:min_shape0].float())
                ).half()
            else: 
                # Standard matrix interpolation: convert to float for precision, then back to half precision (FP16)
                opt["weight"][key] = (
                    ratio * (ckpt1[key].float()) + (1 - ratio) * (ckpt2[key].float())
                ).half()

        # Re-attach original metadata properties to the fused dictionary
        opt["config"] = cfg
        opt["sr"] = cfg_sr
        opt["f0"] = cfg_f0
        opt["version"] = cfg_version
        opt["vocoder"] = vocoder
        # Store detailed tracking info regarding the fusion process parameters
        opt["infos"] = translations["model_fushion_info"].format(
            name=name, 
            model_path_1=model_path_1, 
            model_path_2=model_path_2, 
            ratio=ratio
        )

        # Prepare directory path and save the unified model checkpoint
        output_model = configs["weights_path"]
        if not os.path.exists(output_model): os.makedirs(output_model, exist_ok=True)

        torch.save(opt, os.path.join(output_model, name))
        # Free memory explicitly
        del ckpt1, ckpt2, opt

        gr_info(translations["success"])
        return [translations["success"], os.path.join(output_model, name)]
    except Exception as e:
        gr_error(message=translations["error_occurred"].format(e=e))
        return [e, None]

def fushion_model(
    modelname, 
    model_path_1, 
    model_path_2, 
    ratio
):
    """
    Wrapper function that validates input names and formats before initiating fusion.

    Args:
        modelname (str): Chosen name for the output file.
        model_path_1 (str): File path for model 1.
        model_path_2 (str): File path for model 2.
        ratio (float): Fusion weight ratio.

    Returns:
        list: Results from `fushion_model_pth` or status logs.
    """

    # Enforce that a name must be provided for saving the output
    if not modelname:
        gr_warning(translations["provide_name_is_save"]) 
        return [translations["provide_name_is_save"], None]

    # Ensure both inputs are standard PyTorch weight files (.pth)
    if model_path_1.endswith(".pth") and model_path_2.endswith(".pth"): 
        return fushion_model_pth(
            modelname, 
            model_path_1, 
            model_path_2, 
            ratio
        )
    else:
        gr_warning(translations["format_not_valid"])
        return [None, None]
    
def onnx_export(model_path, is_half, int8_mode):    
    """
    Exports a PyTorch RVC model (.pth) to the ONNX format.

    Args:
        model_path (str): Path to the source .pth model.
        is_half (bool): If True, exports the model using FP16 precision.
        int8_mode (bool): If True, applies INT8 quantization optimizations.

    Returns:
        str: The returned output status or object generated by the underlying `onnx_exporter`.
    """

    # Handle filename auto-completion and configuration directory joining
    if not model_path.endswith(".pth"): model_path += ".pth"
    model_path = os.path.join(configs["weights_path"], model_path) if not os.path.exists(model_path) else model_path

    # Validate source path existence
    if (
        not model_path or 
        not os.path.exists(model_path) or 
        not model_path.endswith(".pth")
    ): 
        return gr_warning(translations["provide_model"])
    
    try:
        gr_info(translations["start_onnx_export"])

        # Deferred dynamic import to keep initialization speed fast
        from main.library.onnx.onnx_export import onnx_exporter

        # Execute ONNX compilation targeting the application's processing device
        output = onnx_exporter(
            model_path, 
            model_path.replace(".pth", ".onnx"), 
            is_half=is_half,
            int8_mode=int8_mode, 
            device=config.device
        )

        gr_info(translations["success"])
        return output
    except Exception as e:
        return gr_error(e)
    
def svc_export(model_path, config_path, modelname, delete_when_success, is_half): 
    """
    Converts/Extracts an So-Vits-SVC model format into an RVC-compatible checkpoint layout.

    Args:
        model_path (str): Path to the source SVC checkpoint (often with a .pth extension).
        config_path (str): Path to the source SVC configuration .json file.
        modelname (str): Final target name for the output RVC model.
        delete_when_success (bool): If True, safely deletes the source SVC files after successful conversion.
        is_half (bool): Use half-precision (FP16) formatting for the output model if checked.

    Returns:
        str: Output tracking information, or None if the input is not a valid SVC model or on failure.
    """

    # Format and check path existence metrics  
    if not model_path.endswith(".pth"): model_path += ".pth"
    model_path = os.path.join(configs["weights_path"], model_path) if not os.path.exists(model_path) else model_path

    if (
        not model_path or 
        not os.path.exists(model_path) or 
        not model_path.endswith(".pth")
    ): 
        gr_warning(translations["provide_model"])
        return None
    
    # Validate the structural layout configuration file (.json) from SVC
    if (
        not config_path or
        not os.path.exists(config_path) or
        not config_path.endswith(".json")
    ):
        gr_warning(translations["provide_config"])
        return None

    # Derive base filename parameters if not specified explicitly
    if modelname is None: modelname = os.path.basename(model_path)
    if not modelname.endswith(".pth"): modelname += ".pth"
    output_path = os.path.join(configs["weights_path"], modelname) if not os.path.exists(modelname) else modelname
    
    try:
        gr_info(translations["start_convert_svc"])

        # Dynamic import of the extraction/conversion tool
        from main.tools.export_svc import svc_converter

        # Execute conversion from SVC weights structure to RVC weights structure
        output = svc_converter(
            model_path, 
            config_path,
            output_path, 
            is_half=is_half
        )

        # Intercept case where the input model is not actually an SVC model structure
        if output == translations["not_svc"]:
            gr_warning(output)
            return None
        
        # Clean up disk space by eliminating original SVC assets if flag is True
        if delete_when_success: os.remove(model_path); os.remove(config_path)
        gr_info(translations["success"])
        return output
    except Exception as e:
        return gr_error(e)

def prettify_date(date_str):
    """Standardizes ISO formatted timestamps into a readable date string format.

    Args:
        date_str (str): Input string containing an ISO timestamp (e.g., '2026-07-12T15:23:45.000').

    Returns:
        str: A clean, human-readable date format string ('%Y-%m-%d %H:%M:%S').
    """

    if date_str is None: 
        return translations["not_found_create_time"]

    try:
        # Parse dynamic precision timestamp fractions and standardize structure output
        return datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        logger.debug(e)
        return translations["format_not_valid"]

def model_info(model_path):
    """
    Reads, extracts, and parses metadata fields contained within a .pth or .onnx model file.

    Args:
        model_path (str): File path of the target model to inspect.

    Returns:
        str: A formatted block containing comprehensive metadata stats of the target model.
    """

    # Resolve directory paths against standard internal structural keys
    model_path = os.path.join(configs["weights_path"], model_path) if not os.path.exists(model_path) else model_path
    model_data = None

    # Ensure targeted file path points directly to a valid data asset
    if (
        not model_path or 
        not os.path.exists(model_path) or 
        os.path.isdir(model_path) or 
        not model_path.endswith((".pth", ".onnx"))
    ): 
        return gr_warning(translations["provide_model"])
    
    gr_info(translations["read_info"])
    # Extract info fields cleanly from internal PyTorch dictionary states
    if model_path.endswith(".pth"): model_data = torch.load(model_path, map_location="cpu")
    else:   
        # Parse embedded configuration structures directly out of ONNX metadata property lists     
        import onnx

        model = onnx.load(model_path)
        for prop in model.metadata_props:
            if prop.key == "model_info":
                model_data = json.loads(prop.value)
                break
        
        del model

    # Normalize variable key names between alternate dataset standard keys
    epochs = model_data.get("epoch", None)
    if not epochs: epochs = model_data.get("info", None)
    if not epochs: epochs = translations["not_found"].format(name="")

    # Render a structural text visualization matching standard system localization configs
    info = translations["model_info"].format(
        model_name=model_data.get("model_name", translations["unregistered"]), 
        model_author=model_data.get("author", translations["not_author"]), 
        epochs=epochs, 
        steps=model_data.get("step", translations["not_found"].format(name="")), 
        version=model_data.get("version", "v1").upper(), 
        sr=model_data.get("sr", translations["not_found"].format(name="")), 
        pitch_guidance=translations["trained_f0" if model_data.get("f0", False) else "not_f0"], 
        model_hash=model_data.get("model_hash", translations["not_found"].format(name="")), 
        creation_date_str=prettify_date(model_data.get("creation_date", None)), 
        vocoder=model_data.get("vocoder", "Default"), 
        speakers_id=model_data.get("speakers_id", 1),
        architecture=model_data.get("architecture", "RVC")
    )

    gr_info(translations["success"])
    del model_data

    return info