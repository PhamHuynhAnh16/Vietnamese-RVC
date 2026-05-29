from torch import nn
from transformers import HubertModel

class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(
            config.hidden_size, 
            config.classifier_proj_size
        )

    def extract_features(self, source, output_layer=None):
        outputs = self.forward(source, output_hidden_states=True, return_dict=True)
        feats = outputs.last_hidden_state if output_layer is None else outputs.hidden_states[output_layer]

        return [feats]