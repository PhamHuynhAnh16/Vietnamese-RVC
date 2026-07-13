import os
import sys
import torch

import numpy as np
import torch.nn.functional as F

from functools import cached_property
from torch.nn.utils.rnn import pad_sequence

sys.path.append(os.getcwd())

from main.library.speaker_diarization.speechbrain import EncoderClassifier

class SpeechBrainPretrainedSpeakerEmbedding:
    """
    A wrapper class for SpeechBrain's pretrained EncoderClassifier to extract 

    speaker embeddings from audio waveforms, with support for temporal masking.
    """

    def __init__(
        self, 
        embedding, 
        device = None
    ):
        """
        Initializes the SpeechBrain pretrained model wrapper.

        Args:
            embedding (str): Path or HuggingFace source ID of the pretrained embedding model.
            device (Optional[Union[str, torch.device]]): Target execution device (e.g., 'cuda' or 'cpu'). Defaults to CPU if not specified.
        """

        super().__init__()

        self.embedding = embedding
        self.device = device or torch.device("cpu")
        # Load the pretrained SpeechBrain model using hyperparameter files
        self.classifier_ = EncoderClassifier.from_hparams(
            source=self.embedding, 
            run_opts={"device": self.device}
        )

    @cached_property
    def dimension(self):
        """
        Determines the embedding output dimension dynamically using a dummy forward pass.

        Returns:
            int: The size/dimension of the generated speaker embedding vectors.
        """

        # Pass a 1-second dummy tensor (assuming 16kHz) through the encoder to catch the shape
        *_, dimension = self.classifier_.encode_batch(
            torch.rand(1, 16000).to(self.device)
        ).shape

        return dimension

    @cached_property
    def min_num_samples(self):
        """
        Finds the absolute minimum number of audio samples required by the model.

        Uses a binary search strategy to find the lower bound threshold where the 
        forward pass no longer throws a processing RuntimeError.

        Returns:
            int: Minimum sample count threshold.
        """

        with torch.inference_mode():
            # Set search boundaries: between 2 samples and 0.5 seconds of audio
            lower, upper = 2, round(0.5 * self.classifier_.audio_normalizer.sample_rate)
            middle = (lower + upper) // 2

            while lower + 1 < upper:
                try:
                    # Attempt encoder forward pass with a dummy tensor of length 'middle'
                    _ = self.classifier_.encode_batch(torch.randn(1, middle).to(self.device))
                    upper = middle # Successfully processed, try a smaller length
                except RuntimeError:
                    lower = middle # Errored out, sample size is too short

                middle = (lower + upper) // 2

        return upper

    def __call__(self, waveforms, masks = None):
        """
        Extracts speaker embeddings from a batch of waveforms with optional voice activity masks.

        Args:
            waveforms (torch.Tensor): Audio tensor.
            masks (Optional[torch.Tensor], optional): Binary temporal mask tensor. Defaults to None.

        Returns:
            np.ndarray: A 2D NumPy array of shape (batch_size, embedding_dimension) 
                containing the embeddings. Sequences that are too short are populated with NaNs.
        """

        batch_size, num_channels, num_samples = waveforms.shape
        assert num_channels == 1

        # Remove the channel dimension -> shape becomes
        waveforms = waveforms.squeeze(dim=1)

        if masks is None:
            signals = waveforms.squeeze(dim=1)
            # All signals within the batch have the same maximum sequence frame length
            wav_lens = signals.shape[1] * torch.ones(batch_size)
        else:
            batch_size_masks, _ = masks.shape
            assert batch_size == batch_size_masks

            # Interpolate low-res masks to match the exact audio sample length via nearest-neighbor
            imasks = F.interpolate(
                masks.unsqueeze(dim=1), 
                size=num_samples, 
                mode="nearest"
            ).squeeze(dim=1) > 0.5 # Convert to a boolean selection matrix

            # Filter out non-masked regions and pad sequences back to form a uniform tensor block
            signals = pad_sequence([
                waveform[imask].contiguous() 
                for waveform, imask in zip(waveforms, imasks)
            ], batch_first=True)
    
            # Compute total active sample frames per batch item
            wav_lens = imasks.sum(dim=1)

        # Edge Case: If the longest sequence in the batch is still under minimum bounds, abort early
        max_len = wav_lens.max()
        if max_len < self.min_num_samples: return np.nan * np.zeros((batch_size, self.dimension))

        # Identify which individual elements fail the minimal length requirement
        too_short = wav_lens < self.min_num_samples
        # SpeechBrain expects relative lengths scaled between 0.0 and 1.0
        wav_lens = wav_lens / max_len
        # Prevent runtime crashes by assigning a fallback dummy length ratio to short tracks
        wav_lens[too_short] = 1.0

        # Execute SpeechBrain batch forward pass
        embeddings = (
            self.classifier_.encode_batch(
                signals, 
                wav_lens=wav_lens
            ).squeeze(dim=1).cpu().numpy()
        )
        # Overwrite invalid/short sequence entries with NaN flags
        embeddings[too_short.cpu().numpy()] = np.nan
        return embeddings