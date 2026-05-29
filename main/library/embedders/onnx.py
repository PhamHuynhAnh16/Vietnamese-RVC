import torch
import onnxruntime

import numpy as np

class HubertModelONNX:
    def __init__(
        self, 
        embedder_model_path, 
        providers
    ):
        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 3

        self.model = onnxruntime.InferenceSession(
            embedder_model_path, 
            sess_options=sess_options, 
            providers=providers
        )

        self._finalproj = False
        self.final_proj = self._final_proj

    def _final_proj(self, source):
        return source
    
    def extract_features(self, source, output_layer = None):
        device = source.device

        logits = self.model.run(
            [self.model.get_outputs()[0].name, self.model.get_outputs()[1].name], 
            {
                self.model.get_inputs()[0].name: source.float().detach().cpu().numpy(),
                self.model.get_inputs()[1].name: np.array(output_layer, dtype=np.int64),
            }
        )

        return [
            torch.as_tensor(
                logits[int(self._finalproj)], 
                dtype=torch.float32, 
                device=device
            )
        ]