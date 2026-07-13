from torch import nn
from transformers import HubertModel

class HubertModelWithFinalProj(HubertModel):
    """
    A subclass of transformers.HubertModel that adds a final linear projection layer 
    and custom feature extraction capabilities.

    Attributes:
        final_proj (nn.Linear): A linear layer that projects features from the hidden size to the classifier projection size.
    """

    def __init__(self, config):
        """
        Initializes the HubertModelWithFinalProj instance.

        Args:
            config (HubertConfig): The configuration object defining model hyperparameters.
        """

        super().__init__(config)
        # Define a linear projection layer mapping from hidden size to classifier projection size
        self.final_proj = nn.Linear(
            config.hidden_size, 
            config.classifier_proj_size
        )

    def extract_features(self, source, output_layer=None):
        """
        Extracts hidden states/features from a specific layer of the Hubert model.

        Args:
            source (torch.Tensor): Input audio waveform tensor of shape (batch_size, sequence_length).
            output_layer (int, optional): The index of the specific transformer layer from which to extract features. If None, the last hidden state is returned.

        Returns:
            List[torch.Tensor]: A list containing a single tensor of the extracted features.
        """

        # Perform a forward pass, explicitly requesting all hidden states to be returned
        outputs = self.forward(source, output_hidden_states=True, return_dict=True)
        # Select the features based on the requested output layer
        # If output_layer is not specified, default to the very last layer's hidden state
        feats = outputs.last_hidden_state if output_layer is None else outputs.hidden_states[output_layer]
        # Return the extracted features wrapped inside a list
        return [feats]