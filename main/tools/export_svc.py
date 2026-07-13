import os
import gc
import sys
import json
import torch
import hashlib
import datetime

sys.path.append(os.getcwd())

from main.app.variables import translations

def svc_converter(model_path, config_path, output_path, is_half=False):
    """
    Converts a So-VITS-SVC checkpoint and its configuration into a unified deployable model format.

    Args:
        model_path (str): Path to the source PyTorch model file (.pth).
        config_path (str): Path to the JSON configuration file.
        output_path (str): Target path where the converted model will be saved.
        is_half (bool, optional): If True, converts model weights to FP16
          (half precision). Defaults to False.

    Returns:
        str: The output file path if successful, or an error message from
        translations if conversion fails.
    """

    try:
        model_dict, weight = {}, {}
        # Load the source PyTorch model onto CPU (using weights_only=True for security)
        model = torch.load(model_path, map_location="cpu", weights_only=True)

        # Validate that the loaded file contains a valid SVC model structure
        if "model" not in model:
            return translations["not_svc"]

        # Process and filter the state dictionary weights
        for key, value in model["model"].items():
            if torch.is_tensor(value): # Cast precision based on the is_half flag
                value = value.half() if is_half else value.float()

            # Map legacy or specific encoder keys to the standardized layout
            if key.startswith("enc_p.enc_."): weight[key.replace("enc_p.enc_.", "enc_p.encoder.")] = value
            elif key.startswith("f0_decoder."): continue # Exclude f0_decoder weights as they are handled differently during inference
            else: weight[key] = value

        # Read and parse the JSON configuration parameters securely
        config = json.load(open(config_path, "r", encoding="utf-8"))

        # Structure the target model metadata and weights dictionary
        model_dict["weight"] = weight
        model_dict["config"] = [
            config["data"]["filter_length"] // 2 + 1,
            config["train"]["segment_size"] / config["data"]["hop_length"], 
            config["model"]["inter_channels"], 
            config["model"]["hidden_channels"], 
            config["model"]["filter_channels"], 
            config["model"]["n_heads"], 
            config["model"]["n_layers"], 
            config["model"]["kernel_size"], 
            config["model"]["p_dropout"], 
            config["model"]["resblock"], 
            config["model"]["resblock_kernel_sizes"], 
            config["model"]["resblock_dilation_sizes"], 
            config["model"]["upsample_rates"], 
            config["model"]["upsample_initial_channel"], 
            config["model"]["upsample_kernel_sizes"], 
            config["model"]["n_speakers"], 
            config["model"]["gin_channels"], 
            config["data"]["sampling_rate"]
        ]

        # Extract training iteration/epoch info (fallback to 0 if not present)
        epoch = model.get("iteration", 0)

        # Try to extract the step count from the filename if it follows the "G_<step>.pth" pattern
        try:
            step = (
                int(os.path.basename(model_path).replace("G_", "").replace(".pth", ""))
            ) if model_path.startswith("G_") and model_path.endswith(".pth") else epoch
        except:
            step = epoch

        # Populate model metadata fields
        model_dict["epoch"] = f"{epoch}epoch"
        model_dict["step"] = step
        model_dict["sr"] = config["data"]["sampling_rate"]
        model_dict["f0"] = True
        # Determine the model version based on Content Vector / SSL dimensions
        model_dict["version"] = "v2" if config["model"]["ssl_dim"] == 768 else "v1"
        model_dict["creation_date"] = datetime.datetime.now().isoformat()
        # Generate a unique hash identity for this specific model package
        model_dict["model_hash"] = hashlib.sha256(f"{str(weight)} {epoch} {step} {datetime.datetime.now().isoformat()}".encode()).hexdigest()
        model_dict["model_name"] = os.path.basename(output_path).replace(".pth", "")
        model_dict["author"] = ""
        model_dict["vocoder"] = "Default"
        model_dict["speakers_id"] = config["model"]["n_speakers"]
        model_dict["architecture"] = "SVC"

        # Save the structured package back to disk
        torch.save(model_dict, output_path)

        # Explicitly clean up heavy tensors and dictionaries to free up RAM
        del model, model_dict, config, weight
        gc.collect()

        return output_path
    except:
        # Fallback error handling if any unexpected file I/O or parsing error occurs
        return translations["not_svc"]