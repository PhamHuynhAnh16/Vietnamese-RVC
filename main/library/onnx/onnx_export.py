import os
import io
import sys
import onnx
import json
import torch
import onnxslim
import warnings
import traceback

sys.path.append(os.getcwd())

from main.app.variables import logger, config
from main.library.algorithm.synthesizers import Synthesizer, SynthesizerSVC

if not config.debug_mode: warnings.filterwarnings("ignore")

FEATS_LENGTH = 200

def autocast(model):
    """
    This function is designed to allow a model—after being converted to the ONNX format—to accept Float-type inputs even when the model is in Half-precision.

    Args:
        model (torch.nn.Module): The PyTorch model to apply autocasting to.

    Returns:
        torch.nn.Module: The modified model with wrapped forward logic.
    """

    orig_forward = model.forward

    def _forward(*args, **kwargs):
        # Fetch the model's internal data type (e.g., float16 or float32)
        dtype = next(model.parameters()).dtype
        # Cast floating-point tensors dynamically, keeping integer or non-tensor inputs untouched
        return orig_forward(*[a.to(dtype) if isinstance(a, torch.Tensor) and a.dtype.is_floating_point else a for a in args], **kwargs)

    model.forward = _forward
    return model

def quantization_int8(output_path):
    """
    Applies dynamic INT8 quantization to an existing ONNX model to reduce file size 
    and accelerate CPU inference.

    Args:
        output_path (str): File path to the original FP32/FP16 ONNX model.

    Returns:
        str: The path to the quantized INT8 ONNX model, or the original path if failed.
    """

    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # Target path for the newly generated INT8 model
    output_path_int8 = output_path.replace(".onnx", "-int8.onnx")

    try:
        # Pre-process the model by executing symbolic shape inference (required before quantization)
        quant_pre_process(output_path, output_path, skip_symbolic_shape=True)
    except Exception as e:
        logger.debug(traceback.format_exc())
        logger.debug(e)

    # Perform dynamic weights quantization to signed 8-bit integers
    quantize_dynamic(model_input=output_path, model_output=output_path_int8, weight_type=QuantType.QInt8)
    # Return the INT8 path if successfully generated; otherwise fallback to the original path
    return output_path_int8 if os.path.exists(output_path_int8) else output_path

def onnx_exporter(input_path, output_path, is_half=False, int8_mode=False, device="cpu"):
    """
    Exports a trained RVC or SVC synthesizer 
    checkpoint to an optimized ONNX model structure, attaching custom metadata.

    Args:
        input_path (str): File path to the PyTorch checkpoint (.pth).
        output_path (str): File path where the output .onnx model will be stored.
        is_half (bool, optional): If True, converts the model weights to Float16. Defaults to False.
        int8_mode (bool, optional): If True, post-processes the model into INT8. Defaults to False.
        device (str, optional): Computation device ('cpu' or 'cuda'). Defaults to "cpu".
    """

    # Force CPU fallback if CUDA is not requested, or if the system utilizes ZLUDA
    if not device.startswith("cuda") or config.is_zluda: device = "cpu"
    
    # Load checkpoint safely avoiding arbitrary code execution hazards
    cpt = (torch.load(input_path, map_location="cpu", weights_only=True) if os.path.isfile(input_path) else None)
    # Adjust dynamic speaker/embedding layer configuration from the loaded weight matrix
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]

    # Unpack model metadata fields from the checkpoint dictionary with default fallbacks
    (
        model_name, 
        model_author, 
        epochs, steps, 
        version, 
        f0, 
        model_hash, 
        vocoder, 
        creation_date, 
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
        cpt.get("speakers_id", 1),
        cpt.get("architecture", "RVC")
    )

    # Set up basic parameters based on model generation version
    text_enc_hidden_dim = 768 if version == "v2" else 256
    tgt_sr = cpt["config"][-1]

    # Instantiate the correct Model Architecture depending on the model specification
    if architecture == "RVC":
        net_g = Synthesizer(
            *cpt["config"], 
            use_f0=f0, 
            text_enc_hidden_dim=text_enc_hidden_dim, 
            vocoder=vocoder, 
            checkpointing=False, 
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

    # Re-route the forward pass to the dedicated ONNX inference method
    net_g.forward = net_g.onnx_infer
    net_g.load_state_dict(cpt["weight"], strict=False)
    # Switch model to evaluation state, transfer to target device, and assign float precision
    net_g.eval().to(device).to(torch.float16 if is_half else torch.float32)
    net_g.remove_weight_norm()
    del net_g.enc_q
    # Apply global dynamic casting wrapper if working with float16 targets
    if is_half: net_g = autocast(net_g)

    # Initialize mock structural dummy input variables for PyTorch's tracing engine
    phone = torch.rand(1, FEATS_LENGTH, text_enc_hidden_dim).to(device)
    phone_length = torch.LongTensor([FEATS_LENGTH]).to(device)
    sid = torch.LongTensor([0]).to(device)
    rate = torch.FloatTensor([1.0]).to(device)

    args = [phone, phone_length, sid, rate]
    input_names = ["phone", "phone_lengths", "sid", "rate"]

    # Set dynamic axes dimensions to allow flexible batch (B) and sequence (T) runtime evaluation
    dynamic_axes = {
        "phone": {0: "B", 1: "T"},
        "phone_lengths": {0: "B"},
        "sid": {0: "B"},
        "rate": {0: "B"}
    }

    if f0: # Append pitch-related parameters if the model variant relies on F0 fundamental frequency
        pitch = torch.randint(size=(1, FEATS_LENGTH), low=5, high=255).to(device)
        nsff0 = torch.rand(1, FEATS_LENGTH).to(device)

        args += [pitch, nsff0]
        input_names += ["pitch", "nsff0"]
        dynamic_axes.update({
            "pitch": {0: "B", 1: "T"},
            "nsff0": {0: "B", 1: "T"},
        })

    try:
        # Export model graph via byte stream pipeline to perform memory-contained optimization
        with io.BytesIO() as model:
            torch.onnx.export(
                net_g, 
                tuple(args), 
                model, 
                do_constant_folding=True, # Pre-calculate constant nodes at compile time
                opset_version=17, 
                verbose=False, 
                input_names=input_names, 
                output_names=["audio"], 
                dynamic_axes=dynamic_axes,
                dynamo=False # Use classic tracing engine
            )

            # Prune unused node structures, fuse blocks and squeeze overhead via onnxslim
            model = onnxslim.slim(onnx.load_model_from_string(model.getvalue()))
            # Embed structured architecture details back directly into the ONNX model metadata block
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
                            "speakers_id": speakers_id,
                            "architecture": architecture
                        }
                    )
                )
            )

        # Save optimized model graph structure out into persistent disk storage
        onnx.save(model, output_path)
        # Clear heavy model instances from RAM memory
        del net_g, model

        # Route down into dynamic INT8 post-quantization pipeline if selected
        if int8_mode: output_path = quantization_int8(output_path)
        return output_path
    except:
        logger.error(traceback.format_exc())
        return None