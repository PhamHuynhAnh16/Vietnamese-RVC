import os
import sys
import json
import torch
import hashlib
import datetime

sys.path.append(os.getcwd())

from main.app.variables import translations

def svc_converter(model_path, config_path, output_path, is_half=False):
    try:
        model_dict, weight = {}, {}
        model = torch.load(model_path, map_location="cpu", weights_only=True)

        if "model" not in model:
            return translations["not_svc"]

        for key, value in model["model"].items():
            if torch.is_tensor(value): 
                value = value.half() if is_half else value.float()

            if key.startswith("enc_p.enc_."): weight[key.replace("enc_p.enc_.", "enc_p.encoder.")] = value
            elif key.startswith("f0_decoder."): continue
            else: weight[key] = value

        config = json.load(open(config_path, "r", encoding="utf-8"))

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

        epoch = model["iteration"]

        try:
            step = (
                int(os.path.basename(model_path).replace("G_", "").replace(".pth", ""))
            ) if model_path.startswith("G_") and model_path.endswith(".pth") else epoch
        except:
            step = epoch

        model_dict["epoch"] = f"{epoch}epoch"
        model_dict["step"] = step
        model_dict["sr"] = config["data"]["sampling_rate"]
        model_dict["f0"] = True
        model_dict["version"] = "v2" if config["model"]["ssl_dim"] == 768 else "v1"
        model_dict["creation_date"] = datetime.datetime.now().isoformat()
        model_dict["model_hash"] = hashlib.sha256(f"{str(weight)} {epoch} {step} {datetime.datetime.now().isoformat()}".encode()).hexdigest()
        model_dict["model_name"] = os.path.basename(output_path).replace(".pth", "")
        model_dict["author"] = ""
        model_dict["vocoder"] = "Default"
        model_dict["energy"] = False
        model_dict["speakers_id"] = config["model"]["n_speakers"]
        model_dict["architecture"] = "SVC"

        torch.save(model_dict, output_path)
        return output_path
    except:
        return translations["not_svc"]