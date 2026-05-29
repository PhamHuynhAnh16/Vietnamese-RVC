import os
import sys
import torch

sys.path.append(os.getcwd())

class PESTO:
    def __init__(
        self, 
        model_path, 
        step_size=10, 
        reduction="alwa", 
        sample_rate=16000, 
        device=None, 
        providers=None, 
        onnx=False,
        compile_model=False,
        compile_mode=None,
        chunk_size = None
    ):
        self.device = device
        self.step_size = step_size
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        if onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            from main.library.predictors.PESTO.model import PPESTO, Resnet1d
            from main.library.predictors.PESTO.preprocessor import Preprocessor

            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            model = PPESTO(
                Resnet1d(
                    **ckpt["hparams"]["encoder"]
                ), 
                preprocessor=Preprocessor(
                    hop_size=step_size, 
                    sampling_rate=sample_rate, 
                    **ckpt["hcqt_params"]
                ), 
                crop_kwargs=ckpt["hparams"]["pitch_shift"], 
                reduction=reduction or ckpt["hparams"]["reduction"]
            )
            model.load_state_dict(ckpt["state_dict"], strict=False)
            model.to(self.device).eval()
            if compile_model: model = torch.compile(model, mode=compile_mode)
        
        self.model = model
        self.infer = self._infer_onnx if onnx else self._infer_torch

    def compute_f0(self, x):
        with torch.inference_mode():
            assert x.ndim <= 2

            preds, confidence = [], []
            total_samples = x.shape[-1]
            if total_samples <= self.chunk_size: return self.infer(x)

            for i in range(0, total_samples, self.chunk_size):
                pred_chunk, conf_chunk = self.infer(x[i : i + self.chunk_size] if x.ndim == 1 else x[:, i : i + self.chunk_size])
                
                preds.append(pred_chunk)
                confidence.append(conf_chunk)

            return torch.cat(preds, dim=-1), torch.cat(confidence, dim=-1)

    def _infer_onnx(self, x):
        model = self.model.run(
            [self.model.get_outputs()[0].name, self.model.get_outputs()[1].name], 
            {self.model.get_inputs()[0].name: x.cpu().numpy()}
        )
        return torch.tensor(model[0], device=self.device), torch.tensor(model[1], device=self.device)

    def _infer_torch(self, x):
        return self.model(
            x, 
            sr=self.sample_rate, 
            convert_to_freq=True, 
            return_activations=False
        )