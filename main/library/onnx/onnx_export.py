import os
import io
import sys
import onnx
import json
import torch
import onnxslim
import warnings

sys.path.append(os.getcwd())

from main.app.variables import logger, config
from main.library.algorithm.synthesizers import Synthesizer, SynthesizerSVC

if not config.debug_mode: warnings.filterwarnings("ignore")

FEATS_LENGTH = 200

def autocast(model):
    orig_forward = model.forward

    def _forward(*args, **kwargs):
        dtype = next(model.parameters()).dtype
        return orig_forward(*[a.to(dtype) if isinstance(a, torch.Tensor) and a.dtype.is_floating_point else a for a in args], **kwargs)

    model.forward = _forward
    return model

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
    if is_half: net_g = autocast(net_g)

    phone = torch.rand(1, FEATS_LENGTH, text_enc_hidden_dim).to(device)
    phone_length = torch.tensor([FEATS_LENGTH]).long().to(device)
    sid = torch.LongTensor([0]).to(device)

    args = [phone, phone_length]
    input_names = ["phone", "phone_lengths"]
    dynamic_axes = {
        "phone": {0: "B", 1: "T"},
        "phone_lengths": {0: "B"},
    }

    if f0:
        pitch = torch.randint(size=(1, FEATS_LENGTH), low=5, high=255).to(device)
        nsff0 = torch.rand(1, FEATS_LENGTH).to(device)

        args += [pitch, nsff0]
        input_names += ["pitch", "nsff0"]
        dynamic_axes.update({
            "pitch": {0: "B", 1: "T"},
            "nsff0": {0: "B", 1: "T"},
        })
    
    args += [sid]
    input_names += ["sid"]
    dynamic_axes.update({
        "sid": {0: "B"}
    })

    if energy_use:
        energy = torch.rand(1, FEATS_LENGTH).to(device)
        args.append(energy)

        input_names.append("energy")
        dynamic_axes.update({
            "energy": {0: "B", 1: "T"},
        })

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
                dynamic_axes=dynamic_axes,
                dynamo=False
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

        onnx.save(model, output_path)
        return output_path
    except:
        import traceback
        logger.error(traceback.format_exc())

        return None