import os
import io
import sys
import onnx
import json
import torch
import onnxslim
import warnings

sys.path.append(os.getcwd())

from main.app.variables import logger
from main.library.algorithm.synthesizers import Synthesizer, SynthesizerSVC

warnings.filterwarnings("ignore")

FEATS_LENGTH = 200

def onnx_exporter(input_path, output_path, is_half=False, device="cpu"):
    if not device.startswith("cuda") or (torch.cuda.is_available() and torch.cuda.get_device_name().endswith("[ZLUDA]")): device = "cpu"

    cpt = (torch.load(input_path, map_location="cpu", weights_only=True) if os.path.isfile(input_path) else None)
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]

    (
        model_name, 
        model_author, 
        epochs, steps, 
        version, 
        f0, 
        model_hash, 
        vocoder, 
        creation_date, 
        energy_use,
        speakers_id,
        architecture
    ) = (
        cpt.get("model_name", None), 
        cpt.get("author", None), 
        cpt.get("epoch", None), 
        cpt.get("step", None), 
        cpt.get("version", "v1"), 
        cpt.get("f0", 1), 
        cpt.get("model_hash", None), 
        cpt.get("vocoder", "Default"), 
        cpt.get("creation_date", None), 
        cpt.get("energy", False),
        cpt.get("speakers_id", 1),
        cpt.get("architecture", "RVC")
    )

    text_enc_hidden_dim = 768 if version == "v2" else 256
    tgt_sr = cpt["config"][-1]

    if architecture == "RVC":
        net_g = Synthesizer(
            *cpt["config"], 
            use_f0=f0, 
            text_enc_hidden_dim=text_enc_hidden_dim, 
            vocoder=vocoder, 
            checkpointing=False, 
            energy=energy_use,
            onnx=True
        )
    else:
        net_g = SynthesizerSVC(
            *cpt["config"], 
            text_enc_hidden_dim=text_enc_hidden_dim, 
            vocoder=vocoder, 
            checkpointing=False, 
            noise_scale=0.4,
            onnx=True
        )

    net_g.forward = net_g.onnx_infer
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().to(device).to(torch.float16 if is_half else torch.float32)
    net_g.remove_weight_norm()

    phone = torch.rand(1, FEATS_LENGTH, text_enc_hidden_dim).to(device)
    phone_length = torch.tensor([FEATS_LENGTH]).long().to(device)
    sid = torch.LongTensor([0]).to(device)

    args = [phone, phone_length]
    input_names = ["phone", "phone_lengths"]
    dynamic_axes = {"phone": [1]}

    if f0:
        pitch = torch.randint(size=(1, FEATS_LENGTH), low=5, high=255).to(device)
        pitchf = torch.rand(1, FEATS_LENGTH).to(device)

        args += [pitch, pitchf]
        input_names += ["pitch", "pitchf"]
        dynamic_axes.update({"pitch": [1], "pitchf": [1]})
    
    args += [sid]

    if energy_use:
        energy = torch.rand(1, FEATS_LENGTH).to(device)
        args.append(energy)

        input_names.append("energy")
        dynamic_axes.update({"energy": [1]})

    try:
        with io.BytesIO() as model:
            torch.onnx.export(
                net_g, 
                tuple(args), 
                model, 
                do_constant_folding=True, 
                opset_version=17, 
                verbose=False, 
                input_names=input_names, 
                output_names=["audio"], 
                dynamic_axes=dynamic_axes
            )

            model = onnxslim.slim(onnx.load_model_from_string(model.getvalue()))
            model.metadata_props.append(
                onnx.StringStringEntryProto(
                    key="model_info", 
                    value=json.dumps(
                        {
                            "model_name": model_name, 
                            "author": model_author, 
                            "epoch": epochs, 
                            "step": steps, 
                            "version": version, 
                            "sr": tgt_sr, 
                            "f0": f0, 
                            "model_hash": model_hash, 
                            "creation_date": creation_date, 
                            "vocoder": vocoder, 
                            "text_enc_hidden_dim": text_enc_hidden_dim,
                            "energy": energy_use,
                            "speakers_id": speakers_id,
                            "architecture": architecture
                        }
                    )
                )
            )

        if is_half:
            import onnxconverter_common
            model = onnxconverter_common.convert_float_to_float16(model, keep_io_types=True)

        onnx.save(model, output_path)
        return output_path
    except:
        import traceback
        logger.error(traceback.format_exc())

        return None