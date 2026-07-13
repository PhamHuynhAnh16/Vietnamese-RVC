import os
import sys
import torch

sys.path.append(os.getcwd())

from main.app.variables import logger, translations
from main.library.algorithm.residuals import ResidualCouplingBlock
from main.library.algorithm.encoders import TextEncoder, PosteriorEncoder, TextEncoderSVC
from main.library.algorithm.commons import slice_segments, rand_slice_segments, sequence_mask

class Synthesizer(torch.nn.Module):
    """
    Synthesizer module for Voice Conversion (commonly used in RVC inference/training pipelines).
    Transforms source speech representations into target speaker characteristics 
    using a PosteriorEncoder, Flow model, and dynamic Vocoder selections.
    """

    def __init__(
        self, 
        spec_channels, 
        segment_size, 
        inter_channels, 
        hidden_channels, 
        filter_channels, 
        n_heads, 
        n_layers, 
        kernel_size, 
        p_dropout, 
        resblock, 
        resblock_kernel_sizes, 
        resblock_dilation_sizes, 
        upsample_rates, 
        upsample_initial_channel, 
        upsample_kernel_sizes, 
        spk_embed_dim, 
        gin_channels, 
        sr, 
        use_f0, 
        text_enc_hidden_dim=768, 
        vocoder="Default", 
        checkpointing=False, 
        onnx=False, 
        **kwargs
    ):
        """Initializes components of the RVC Synthesizer architecture."""

        super(Synthesizer, self).__init__()
        self.segment_size = segment_size
        self.use_f0 = use_f0

        # Feature/Text Encoder processes linguistic/content units into prior distributions
        self.enc_p = TextEncoder(
            inter_channels, 
            hidden_channels, 
            filter_channels, 
            n_heads, 
            n_layers, 
            kernel_size, 
            float(p_dropout), 
            text_enc_hidden_dim, 
            f0=use_f0, 
            onnx=onnx
        )

        # Dynamic mapping setup for targeted Audio Waveform Generators (Vocoders)
        if use_f0:
            if vocoder == "RefineGAN": 
                from main.library.generators.refinegan import RefineGANGenerator

                logger.info(translations["use_vocoders"].format(name="REFINEGAN"))

                self.dec = RefineGANGenerator(
                    sample_rate=sr, 
                    upsample_rates=upsample_rates, 
                    num_mels=inter_channels, 
                    checkpointing=checkpointing
                )
            elif vocoder == "BigVGAN":
                from main.library.generators.bigvgan import BigVGANGenerator

                logger.info(translations["use_vocoders"].format(name="BIGVGAN"))

                self.dec = BigVGANGenerator(
                    in_channel=inter_channels,
                    upsample_initial_channel=upsample_initial_channel,
                    upsample_rates=upsample_rates,
                    upsample_kernel_sizes=upsample_kernel_sizes,
                    resblock_kernel_sizes=resblock_kernel_sizes,
                    resblock_dilations=resblock_dilation_sizes,
                    gin_channels=gin_channels,
                    sample_rate=sr,
                    harmonic_num=0, 
                )
            elif vocoder in ["MRF-HiFi-GAN", "MRF HiFi-GAN"]: 
                from main.library.generators.mrf_hifigan import HiFiGANMRFGenerator

                logger.info(translations["use_vocoders"].format(name="MRF-HIFIGAN"))

                self.dec = HiFiGANMRFGenerator(
                    in_channel=inter_channels, 
                    upsample_initial_channel=upsample_initial_channel, 
                    upsample_rates=upsample_rates, 
                    upsample_kernel_sizes=upsample_kernel_sizes, 
                    resblock_kernel_sizes=resblock_kernel_sizes, 
                    resblock_dilations=resblock_dilation_sizes, 
                    gin_channels=gin_channels, 
                    sample_rate=sr, 
                    harmonic_num=8, 
                    checkpointing=checkpointing
                )
            else: 
                from main.library.generators.nsf_hifigan import HiFiGANNSFGenerator

                logger.info(translations["use_vocoders"].format(name="NSF-HIFIGAN"))

                self.dec = HiFiGANNSFGenerator(
                    inter_channels, 
                    resblock_kernel_sizes, 
                    resblock_dilation_sizes, 
                    upsample_rates, 
                    upsample_initial_channel, 
                    upsample_kernel_sizes, 
                    gin_channels=gin_channels, 
                    sr=sr, 
                    checkpointing=checkpointing,
                    harmonic_num=0
                )
        else: 
            from main.library.generators.hifigan import HiFiGANGenerator

            logger.info(translations["use_vocoders"].format(name="HIFIGAN"))

            self.dec = HiFiGANGenerator(
                inter_channels, 
                resblock_kernel_sizes, 
                resblock_dilation_sizes, 
                upsample_rates, 
                upsample_initial_channel, 
                upsample_kernel_sizes, 
                gin_channels=gin_channels
            )

        # Posterior Encoder extracts latent traits from target linear spectrograms during training
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        # Normalizing Flow refines complex posteriors into simpler prior distributions invertibly
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)
        # Target Speaker Identity Lookup Table
        self.emb_g = torch.nn.Embedding(spk_embed_dim, gin_channels)

    def remove_weight_norm(self):
        """Strips weight normalization flags to optimize execution performance for production."""

        for module in [self.dec, self.flow, self.enc_q]:
            module.remove_weight_norm()

    @torch.jit.ignore
    def forward(self, phone, phone_lengths, pitch = None, pitchf = None, y = None, y_lengths = None, ds = None):
        """Runs the Variational Inference forward training loop for Voice Conversion."""

        g = self.emb_g(ds).unsqueeze(-1) # Global conditioning
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)

        if y is not None:
            # Extract posterior representation from target speech spectrogram (y)
            z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
            # Map target latents into the flow space
            z_p = self.flow(z, y_mask, g=g)

            # Slice acoustic segments randomly to preserve bounded GPU memory tracking
            z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
            f0_slice = slice_segments(pitchf, ids_slice, self.segment_size, 2) if self.use_f0 else None

            # Generate target converted audio waveform slice
            return (
                self.dec( 
                    z_slice, 
                    f0_slice, 
                    g=g
                ), 
                ids_slice, 
                x_mask, 
                y_mask, 
                (z, z_p, m_p, logs_p, m_q, logs_q)
            )
        else: return None, None, x_mask, None, (None, None, m_p, logs_p, None, None)

    @torch.jit.export
    def infer(self, phone, phone_lengths, pitch = None, nsff0 = None, sid = None, rate = None):
        """Executes source-to-target voice conversion inference within PyTorch."""

        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)

        # Sample latent vectors from source content priors using reparameterization
        z_p = (m_p + logs_p.exp() * torch.randn_like(m_p) * 0.66666) * x_mask

        if rate is not None:
            # Adjusts speech speed/tempo offset allocations
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p, x_mask = z_p[:, :, head:], x_mask[:, :, head:]
            if self.use_f0: nsff0 = nsff0[:, head:]

        # Pass priors through Inverse Flow to reconstruct converted target latents
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        return self.dec(z * x_mask, nsff0, g=g)

    def onnx_infer(self, phone, phone_lengths, sid = None, rate = None, pitch = None, nsff0 = None):
        """Alternative target inference route utilizing ONNX-exportable tracing steps."""

        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)

        # Sample latent vectors from source content priors using reparameterization
        z_p = (m_p + logs_p.exp() * torch.randn_like(m_p) * 0.66666) * x_mask

        # Floor and casting are explicitly typed for static tensor indexing compliance in ONNX runtimes
        head = torch.floor(z_p.size(2) * (1.0 - rate)).long()
        z_p, x_mask = z_p[:, :, head:], x_mask[:, :, head:]
        if self.use_f0: nsff0 = nsff0[:, head:]

        # Pass priors through Inverse Flow to reconstruct converted target latents
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        return self.dec(z * x_mask, nsff0, g=g)

class SynthesizerSVC(torch.nn.Module):
    """
    Synthesizer module customized specifically for Singing Voice Conversion (SVC).
    Processes dense acoustic speech feature vectors (e.g., ContentVec/HuBERT) instead of text tokens
    and synchronizes source expressions with the target singer's voice profile.
    """

    def __init__(
        self, 
        spec_channels, 
        segment_size, 
        inter_channels, 
        hidden_channels, 
        filter_channels, 
        n_heads, 
        n_layers, 
        kernel_size, 
        p_dropout, 
        resblock, 
        resblock_kernel_sizes, 
        resblock_dilation_sizes, 
        upsample_rates, 
        upsample_initial_channel, 
        upsample_kernel_sizes, 
        spk_embed_dim, 
        gin_channels, 
        sr, 
        text_enc_hidden_dim=768, 
        vocoder="Default", 
        checkpointing=False, 
        onnx=False, 
        noise_scale=0.35,
        **kwargs
    ):
        """Initializes architectural modules for Singing Voice Conversion pipelines."""

        super().__init__()
        self.segment_size = segment_size
        self.noise_scale = noise_scale
        self.sr = sr
        # Content/Acoustic feature representation encoder
        self.enc_p = TextEncoderSVC(
            inter_channels,
            hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            onnx=onnx
        )

        # Dynamic mapping setup for targeted Audio Waveform Generators (Vocoders)
        if vocoder == "RefineGAN": 
            from main.library.generators.refinegan import RefineGANGenerator

            logger.info(translations["use_vocoders"].format(name="REFINEGAN"))

            self.dec = RefineGANGenerator(
                sample_rate=sr, 
                upsample_rates=upsample_rates, 
                num_mels=inter_channels, 
                checkpointing=checkpointing
            )
        elif vocoder == "BigVGAN":
            from main.library.generators.bigvgan import BigVGANGenerator

            logger.info(translations["use_vocoders"].format(name="BIGVGAN"))

            self.dec = BigVGANGenerator(
                in_channel=inter_channels,
                upsample_initial_channel=upsample_initial_channel,
                upsample_rates=upsample_rates,
                upsample_kernel_sizes=upsample_kernel_sizes,
                resblock_kernel_sizes=resblock_kernel_sizes,
                resblock_dilations=resblock_dilation_sizes,
                gin_channels=gin_channels,
                sample_rate=sr,
                harmonic_num=0, 
            )
        elif vocoder in ["MRF-HiFi-GAN", "MRF HiFi-GAN"]: 
            from main.library.generators.mrf_hifigan import HiFiGANMRFGenerator

            logger.info(translations["use_vocoders"].format(name="MRF-HIFIGAN"))

            self.dec = HiFiGANMRFGenerator(
                in_channel=inter_channels, 
                upsample_initial_channel=upsample_initial_channel, 
                upsample_rates=upsample_rates, 
                upsample_kernel_sizes=upsample_kernel_sizes, 
                resblock_kernel_sizes=resblock_kernel_sizes, 
                resblock_dilations=resblock_dilation_sizes, 
                gin_channels=gin_channels, 
                sample_rate=sr, 
                harmonic_num=8, 
                checkpointing=checkpointing
            )
        else: 
            from main.library.generators.nsf_hifigan import HiFiGANNSFGenerator

            logger.info(translations["use_vocoders"].format(name="NSF-HIFIGAN"))

            self.dec = HiFiGANNSFGenerator(
                inter_channels, 
                resblock_kernel_sizes, 
                resblock_dilation_sizes, 
                upsample_rates, 
                upsample_initial_channel, 
                upsample_kernel_sizes, 
                gin_channels=gin_channels, 
                sr=sr, 
                checkpointing=checkpointing,
                harmonic_num=8
            )

        # Discrete Voiced/Unvoiced parameter tracking table
        self.emb_uv = torch.nn.Embedding(2, hidden_channels)
        self.emb_g = torch.nn.Embedding(spk_embed_dim, gin_channels)
        # Projection layer to map external content feature vectors into hidden channels
        self.pre = torch.nn.Conv1d(text_enc_hidden_dim, hidden_channels, kernel_size=5, padding=2)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

    def remove_weight_norm(self):
        """Removes weight normalization flags across submodules."""

        for module in [self.dec, self.flow, self.enc_q]:
            module.remove_weight_norm()

    def forward(self, phone, phone_lengths, pitch = None, pitchf = None, y = None, y_lengths = None, ds = None):
        """Runs the SVC forward training loop using content representations."""

        g = self.emb_g(ds.unsqueeze(0) if ds.dim() == 1 else ds).transpose(1, 2)
        phone = phone.transpose(1, 2) # Reshape layout to channel-first format

        x_mask = sequence_mask(phone_lengths, phone.size(2)).unsqueeze(1).to(phone.dtype)
        # Combine projected content embeddings with fundamental voiced/unvoiced sequence flags
        x = self.pre(phone) * x_mask + self.emb_uv((pitchf > 0.0).long()).transpose(1, 2)

        _, m_p, logs_p, _ = self.enc_p(x, x_mask, f0=pitch)
        z, m_q, logs_q, spec_mask = self.enc_q(y, y_lengths, g=g)

        z_p = self.flow(z, spec_mask, g=g)
        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)

        return (
            self.dec(
                z_slice, 
                g=g, 
                f0=slice_segments(pitchf, ids_slice, self.segment_size, 2)
            ), 
            ids_slice, 
            x_mask, 
            spec_mask, 
            (z, z_p, m_p, logs_p, m_q, logs_q)
        )

    @torch.no_grad()
    def infer(self, phone, phone_lengths, pitch = None, nsff0 = None, sid = None, rate = None):
        """Transforms source vocal content into targeted singer outputs during SVC inference."""

        g = self.emb_g(sid.unsqueeze(0) if sid.dim() == 1 else sid).transpose(1, 2)
        phone = phone.transpose(1, 2)
        
        x_mask = sequence_mask(phone_lengths, phone.size(2)).unsqueeze(1).to(phone.dtype)
        x = self.pre(phone) * x_mask + self.emb_uv((nsff0 > 0.0).long()).transpose(1, 2)

        # Extract content prior conditions adjusted via noise scale factors
        z_p, _, _, c_mask = self.enc_p(x, x_mask, f0=pitch, noise_scale=self.noise_scale)

        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p, c_mask, nsff0 = z_p[:, :, head:], c_mask[:, :, head:], nsff0[:, head:]

        # Convert priors to singer latents through the Inverse Flow block
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        o = self.dec(z * c_mask, g=g, f0=nsff0)

        # Linearly interpolate audio duration dimensions if the output length drifts from hop targets
        target_len = phone_lengths * (self.sr // 100)
        if o.shape[-1] != target_len:
            o = torch.nn.functional.interpolate(
                o,
                size=target_len,
                mode="linear",
                align_corners=False
            )

        return o

    @torch.no_grad()
    def onnx_infer(self, phone, phone_lengths, sid = None, rate = None, pitch = None, nsff0 = None):
        """ONNX-friendly singing voice conversion route utilizing tracking tracing operators."""

        g = self.emb_g(sid.unsqueeze(0) if sid.dim() == 1 else sid).transpose(1, 2)
        phone = phone.transpose(1, 2)
        
        x_mask = sequence_mask(phone_lengths, phone.size(2)).unsqueeze(1).to(phone.dtype)
        x = self.pre(phone) * x_mask + self.emb_uv((nsff0 > 0.0).long()).transpose(1, 2)

        # Extract content prior conditions adjusted via noise scale factors
        z_p, _, _, c_mask = self.enc_p(x, x_mask, f0=pitch, noise_scale=self.noise_scale)

        head = torch.floor(z_p.size(2) * (1.0 - rate)).long()
        z_p, c_mask, nsff0 = z_p[:, :, head:], c_mask[:, :, head:], nsff0[:, head:]

        # Convert priors to singer latents through the Inverse Flow block
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        o = self.dec(z * c_mask, g=g, f0=nsff0)

        # Linearly interpolate audio duration dimensions if the output length drifts from hop targets
        target_len = phone_lengths * (self.sr // 100)
        if o.shape[-1] != target_len:
            o = torch.nn.functional.interpolate(
                o,
                size=target_len,
                mode="linear",
                align_corners=False
            )

        return o