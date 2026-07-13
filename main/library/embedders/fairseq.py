import os
import re
import sys
import math
import uuid
import torch
import types
import contextlib

import numpy as np
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from torch import nn
from omegaconf import DictConfig, open_dict

sys.path.append(os.getcwd())

from main.library.algorithm.normalization import Fp32GroupNorm, Fp32LayerNorm

def load_model(filename):
    """
    Loads a Hubert model checkpoint from a specified file path.

    It attempts a safe load first. If it fails, it checks configuration 
    fallbacks to allow unsafe fairseq dictionary loading before initializing 
    the architecture.

    Args:
        filename (str): Path to the model checkpoint file.

    Returns:
        HubertModel: The initialized HuBERT model with loaded weights.
    """

    try:
        # Attempt secure loading of weights using restricted unpickling (weights_only=True)
        # to block arbitrary code execution risks embedded within malicious pickles.
        state = torch.load(filename, map_location="cpu", weights_only=True)
    except:
        # Fallback tracking if the secure loading fails, typically because fairseq structures require full module definitions to unpickle.
        from main.app.variables import logger, configs, translations

        logger.warning(translations["hubert_warn_1"])
        logger.warning(translations["hubert_warn_2"])

        # Check application runtime configurations to see if the user explicitly allowed unsafe loads
        if configs.get("allow_unsafe_fairseq", False):
            # Define an empty mock class to step in for the missing fairseq Dictionary structure
            class Dictionary:
                def __init__(self, *args, **kwargs):
                    pass

            # Dynamically instantiate mock modules and hook them directly into sys.modules.
            # This fools the pickle unpickler into believing fairseq is fully installed.
            fairseq = types.ModuleType("fairseq")
            fairseq_data = types.ModuleType("fairseq.data")
            fairseq_data_dictionary = types.ModuleType("fairseq.data.dictionary")
            # Map the mock Dictionary class onto our dynamically spawned namespace structure
            fairseq_data_dictionary.Dictionary = Dictionary
            fairseq.data = fairseq_data
            fairseq_data.dictionary = fairseq_data_dictionary
            # Register the dynamic modules in global system cache so internal torch.load calls match them
            sys.modules["fairseq"] = fairseq
            sys.modules["fairseq.data"] = fairseq_data
            sys.modules["fairseq.data.dictionary"] = fairseq_data_dictionary

            # Re-attempt the load with weights_only=False, allowing standard (potentially unsafe) unpickling
            state = torch.load(filename, map_location="cpu", weights_only=False)
        else: sys.exit(0)

    # Instantiate the target model architecture using configurations parsed directly out of the checkpoint dictionary
    # The number of target output classes is determined dynamically by viewing the shape of the label embeddings.
    model = HubertModel(
        HubertConfig(
            **state['cfg']['model']
        ), 
        num_classes=int(state['model']['label_embs_concat'].shape[0])
    )
    # Inject state dictionary weights into model parameters; strict=False ignores missing/extra non-critical keys
    model.load_state_dict(state['model'], strict=False)
    del state

    return model

def softmax(x, dim, onnx_trace = False):
    """Computes softmax activation over a specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): The dimension along which softmax will be computed.
        onnx_trace (bool, optional): If True, casts tensor to float before softmax 
            to ensure ONNX compatibility. Defaults to False.

    Returns:
        torch.Tensor: Normalized probability distribution.
    """

    # ONNX tracking has strict constraints regarding data type inference tracking.
    # If tracing is active, explicitly cast to float, otherwise execute native float32-enforced activation.
    return F.softmax(x.float(), dim=dim) if onnx_trace else F.softmax(x, dim=dim, dtype=torch.float32)

def log_softmax(x, dim, onnx_trace = False):
    """Computes log-softmax activation over a specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): The dimension along which log-softmax will be computed.
        onnx_trace (bool, optional): If True, casts tensor to float before log-softmax 
            to ensure ONNX compatibility. Defaults to False.

    Returns:
        torch.Tensor: Log-probabilities tensor.
    """

    # Follows identical pattern as softmax helper: handles ONNX graph export limitations securely.
    return F.log_softmax(x.float(), dim=dim) if onnx_trace else F.log_softmax(x, dim=dim, dtype=torch.float32)

def with_incremental_state(cls):
    """Class decorator to seamlessly inject FairseqIncrementalState capabilities into a class hierarchy.

    Args:
        cls (type): The target class to be decorated.

    Returns:
        type: Modified class with FairseqIncrementalState as its primary base class.
    """

    # Manipulate Python MRO (Method Resolution Order) tuple dynamically by prefixing the target class base list
    # with the FairseqIncrementalState class object, eliminating redundant inheritances along the way.
    cls.__bases__ = (FairseqIncrementalState,) + tuple(b for b in cls.__bases__ if b != FairseqIncrementalState)
    return cls

def quant_noise(module, p, block_size):
    """Applies quantization noise regularization to a module's weights during training.

    Args:
        module (nn.Module): Linear, Embedding, or Conv2d layer to apply noise to.
        p (float): Probability of introducing quantization noise (dropout rate for blocks).
        block_size (int): Size of structural blocks to apply noise quantization to.

    Returns:
        nn.Module: The modified module attached with a forward pre-hook for quantization noise.
    """

    # Short-circuit out immediately if the quantization execution probability is zero or negative
    if p <= 0: return module
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))
    is_conv = module.weight.ndim == 4

    if not is_conv: 
        # Linear/Embedding check: Input dimensions must be cleanly divisible by the block clustering rate
        assert (module.weight.size(1) % block_size == 0)
    else:
        if module.kernel_size == (1, 1): 
            # 1x1 Convolution check: Total input channels must be divisible by block groups
            assert (module.in_channels % block_size == 0)
        else:
            # Spatial Conv check: The product of kernel height and width parameters must align with block sizes
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0

    def _forward_pre_hook(mod, input):
        """Applies random block-based zeroing-out noise to model parameters during training."""

        # Quantization noise injection is exclusively active during standard model training mode iterations
        if mod.training:
            if not is_conv:
                weight = mod.weight

                in_features = weight.size(1)
                out_features = weight.size(0)
                # Initialize flat zeros array corresponding to target scaled feature blocks
                mask = torch.zeros(in_features // block_size * out_features, device=weight.device)
                # Populate values randomly using a Bernoulli trials distribution set at execution probability 'p'
                mask.bernoulli_(p)
                # Expand mask back out across block windows and reshape to match structural weight rank configurations
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)
            else:
                weight = mod.weight

                in_channels = mod.in_channels
                out_channels = mod.out_channels

                if mod.kernel_size == (1, 1):
                    # Construct and map block masks specific to flat 1x1 convolutional tensors
                    mask = torch.zeros(int(in_channels // block_size * out_channels), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    # Construct structural spatial masks spanning cross-channel dimension planes entirely
                    mask = torch.zeros(weight.size(0), weight.size(1), device=weight.device)
                    mask.bernoulli_(p)
                    # Extend matrix ranks via unsqueeze operators to align seamlessly with spatial
                    mask = (mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1]))

            # Convert mask elements to boolean tensors for efficient masking masking behaviors
            mask = mask.to(torch.bool)
            # Compute inverted scaling coefficient factor to keep overall parameter magnitude distributions consistent
            s = 1 / (1 - p)
            # Mutate weight data in place, filling selected masked regions with zero and scaling active values
            mod.weight.data = s * weight.masked_fill(mask, 0)

    # Attach our defined hook to run immediately prior to executing standard forward layers processing passes
    module.register_forward_pre_hook(_forward_pre_hook)
    return module

class FairseqDropout(nn.Module):
    """Dropout wrapper designed to allow configuration of dropout rules during inference mode."""

    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p # Target probability field configuration parameter
        self.module_name = module_name # Internal descriptor key tag used to isolate modules for targeted configuration changes
        self.apply_during_inference = False # State flag allowing active dropping evaluations during test loops

    def forward(self, x, inplace = False):
        # Apply dropping function if execution rate is positive AND module status matches active generation metrics
        return F.dropout(x, p=self.p, training=True, inplace=inplace) if self.p > 0 and (self.training or self.apply_during_inference) else x

    def make_generation_fast_(self, name, retain_dropout = False, retain_dropout_modules = None, **kwargs):
        """Enables specific dropout layers during inference tracking to support variational generation."""

        # If tracking state overrides are globally called, check if our local layer identifier matches explicit targeted lists
        if retain_dropout and (retain_dropout_modules is None or self.module_name in retain_dropout_modules): self.apply_during_inference = True

class FairseqIncrementalState(object):
    """Abstract interface class providing standard handling for incremental decoding buffers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        # Generate an isolated completely distinct uuid sequence string to securely index local module storage items
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key):
        # Constructs fully qualified compound unique mapping keys to safeguard against dictionary key collision risks
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(self, incremental_state, key):
        """Extracts cached decoder states based on specific unique context keys."""

        full_key = self._get_full_incremental_state_key(key)
        # Check dictionary records safely; if tracking object isn't constructed, gracefully output default None values
        if incremental_state is None or full_key not in incremental_state: return None
        return incremental_state[full_key]

    def set_incremental_state(self, incremental_state, key, value):
        """Caches structural sequence tensors into the active global incremental state storage."""

        # Mutate the parent shared inference context track matrix in-place by writing current state records
        if incremental_state is not None: incremental_state[self._get_full_incremental_state_key(key)] = value
        return incremental_state

class FairseqDecoder(nn.Module):
    """Base abstract decoder structure utilized within sequence translation architectures."""

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary
        self.onnx_trace = False # Operational flag handling active standard deployment parsing graphs
        self.adaptive_softmax = None # Placeholder attribute mapping hierarchical projection functions

    def forward(self, prev_output_tokens, encoder_out=None, **kwargs):
        """Processes target sequences through features layer mapping up to final logit evaluations."""
        # Route historical tracking input structures down through primary mapping layer pathways
        x, extra = self.extract_features(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        # Push abstract sequence maps through final projection blocks to map vocabulary dimensions
        return self.output_layer(x), extra

    def extract_features(self, prev_output_tokens, encoder_out=None, **kwargs):
        # Abstract placeholder design; must be declared inside inheriting sub-architectures
        pass

    def output_layer(self, features, **kwargs):
        # Abstract placeholder design; must be declared inside inheriting sub-architectures
        pass

    def get_normalized_probs(self, net_output, log_probs, sample = None):
        """Retrieves normalized predictive distribution tensor outputs."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    def get_normalized_probs_scriptable(self, net_output, log_probs, sample = None):
        """Computes softmax probabilities or log-probabilities from model outputs in a scriptable format."""

        # Verify if an adaptive projection calculation alternative is explicitly deployed in place
        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else: target = None
            
            # Route tokens down specialized adaptive tracking pipelines to compute regional target distributions
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            # If log distribution metrics are rejected by call parameters, unpack exponentials to get standard probabilities
            return out.exp_() if not log_probs else out

        # Fallback tracking sequence route: compute standard calculations using standard full vocabulary structures
        logits = net_output[0]
        return log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace) if log_probs else softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

    def max_positions(self):
        """Returns the maximum absolute sequence length threshold."""

        return 1e6

    def upgrade_state_dict_named(self, state_dict, name):
        """Hook method to adjust parameter naming compatibility across checkpoint versions."""

        return state_dict

    def prepare_for_onnx_export_(self):
        """Flips processing flags to conform to standard tracking limitations under ONNX compilation."""

        self.onnx_trace = True

@with_incremental_state
class FairseqIncrementalDecoder(FairseqDecoder):
    """Extension of FairseqDecoder explicitly supporting sequential state caching (incremental decoding)."""

    def __init__(self, dictionary):
        super().__init__(dictionary)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        pass

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        pass

    def reorder_incremental_state(self, incremental_state, new_order):
        pass

    def reorder_incremental_state_scripting(self, incremental_state, new_order):
        """Reorders cached hidden states to sync up with changed step exploration paths during beam search."""

        # Traverse recursively down through sub-modules to update index arrangements within caching tracking buffers
        for module in self.modules():
            if hasattr(module, "reorder_incremental_state"):
                result = module.reorder_incremental_state(incremental_state, new_order)
                if result is not None: incremental_state = result

    def set_beam_size(self, beam_size):
        """Sets internal search beam tracking attributes dynamically across sub-modules."""

        if getattr(self, "_beam_size", -1) != beam_size:
            seen = set()
            # Local helper mapping utility tracking structural execution history paths
            def apply_set_beam_size(module):
                if (module != self and hasattr(module, "set_beam_size") and module not in seen):
                    seen.add(module)
                    module.set_beam_size(beam_size)

            # Route functional application mappings down full nested network trees completely
            self.apply(apply_set_beam_size)
            self._beam_size = beam_size

class MultiheadAttention(FairseqIncrementalDecoder):
    """Multi-Head Attention module supporting self-attention, cross-attention, and incremental caching."""

    def __init__(
        self, 
        embed_dim, 
        num_heads, 
        kdim=None, 
        vdim=None, 
        dropout=0.0, 
        bias=True, 
        add_bias_kv=False, 
        add_zero_attn=False, 
        self_attention=False, 
        encoder_decoder_attention=False, 
        dictionary=None, 
        q_noise=0.0, 
        qn_block_size=8, 
        xformers_att_config=None, 
        xformers_blocksparse_layout=None, 
        xformers_blocksparse_blocksize=16
    ):
        super().__init__(dictionary)
        if xformers_att_config is None: xformers_att_config = None
        # Evaluate configuration script strings dynamically if configurations are encapsulated inside flat text format blocks
        if isinstance(xformers_att_config, str): xformers_att_config = eval(xformers_att_config)
        # Parse boolean metrics indicating if highly optimized external xformers execution libraries are demanded
        self.use_xformers = xformers_att_config is not None
        if self.use_xformers: raise ImportError("Configuration mismatch: xformers acceleration backend called but unsupported in current architecture environments.")
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        # Check if projection shapes map symmetrically to determine parameter initialization options later
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        # Initialize isolated structural dropout tracking blocks
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        # Compute specific inner hidden dimension metrics assigned uniquely per core attention head calculation channel
        self.head_dim = embed_dim // num_heads
        # Assert head divisibility compatibility directly to avoid bad matrix mapping shape states during forward actions
        assert (self.head_dim * num_heads == self.embed_dim)
        # Calculate scaled dot-product attention stabilization multiplier factor
        self.scaling = self.head_dim**-0.5
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim
        # Initialize linear weight projection components wrapped with quantization noise simulation masks
        self.k_proj = quant_noise(nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.v_proj = quant_noise(nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.q_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        # Handle custom permanent context parameter embeddings configurations if spatial biases are requested
        if add_bias_kv: self.bias_k, self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim)), nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else: self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self.beam_size = 1
        self.reset_parameters()
        self.onnx_trace = False
        self.skip_embed_dim_check = False
        self.init_incremental_state()

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        """Initializes model weights using Xavier uniform initialization."""

        if self.qkv_same_dim:
            # Scale gain factors effectively down to compensate for overlapping tracking channels variances
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        # Set default zero parameters for bias maps safely
        if self.out_proj.bias is not None: nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None: nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None: nn.init.xavier_normal_(self.bias_v)

    def _get_reserve_head_index(self, num_heads_to_keep: int):
        """Identifies and retains index pointers for attention heads exhibiting highest weight norms."""

        k_proj_heads_norm, q_proj_heads_norm, v_proj_heads_norm = [], [], []
        # Iterate over separate heads tracking blocks to extract aggregate internal matrix norm summaries
        for i in range(self.num_heads):
            start_idx = i * self.head_dim
            end_idx = (i + 1) * self.head_dim

            # Sum up absolute weights and bias records belonging uniquely inside the active head segment slice
            k_proj_heads_norm.append(
                (self.k_proj.weight[start_idx:end_idx]).abs().sum().tolist() + 
                (self.k_proj.bias[start_idx:end_idx]).abs().sum().tolist()
            )
            q_proj_heads_norm.append(
                (self.q_proj.weight[start_idx:end_idx]).abs().sum().tolist() + 
                (self.q_proj.bias[start_idx:end_idx]).abs().sum().tolist()
            )
            v_proj_heads_norm.append(
                (self.v_proj.weight[start_idx:end_idx]).abs().sum().tolist() + 
                (self.v_proj.bias[start_idx:end_idx]).abs().sum().tolist()
            )

        # Combine computed projection statistics across matrix families per individual head indices
        heads_norm = []
        for i in range(self.num_heads):
            heads_norm.append(k_proj_heads_norm[i] + q_proj_heads_norm[i] + v_proj_heads_norm[i])
        # Sort head reference index markers from maximum to minimum depending on importance scores
        sorted_head_index = sorted(range(self.num_heads), key=lambda k: heads_norm[k], reverse=True)
        # Isolate boundary mapping coordinate pairs for the top-N retained structural attention tracks
        reserve_head_index = []
        for i in range(num_heads_to_keep):
            reserve_head_index.append((sorted_head_index[i] * self.head_dim, (sorted_head_index[i] + 1) * self.head_dim))

        return reserve_head_index

    def _adaptive_prune_heads(self, reserve_head_index):
        """Prunes less informative attention heads structurally from projection weights."""

        new_q_weight, new_q_bias, new_k_weight, new_k_bias, new_v_weight, new_v_bias, new_out_proj_weight = [], [], [], [], [], [], []
        # Slice relevant sections out of weight blocks using coordinates collected from the validation index maps
        for ele in reserve_head_index:
            start_idx, end_idx = ele

            new_q_weight.append(self.q_proj.weight[start_idx:end_idx])
            new_q_bias.append(self.q_proj.bias[start_idx:end_idx])

            new_k_weight.append(self.k_proj.weight[start_idx:end_idx])
            new_k_bias.append(self.k_proj.bias[start_idx:end_idx])

            new_v_weight.append(self.v_proj.weight[start_idx:end_idx])
            new_v_bias.append(self.v_proj.bias[start_idx:end_idx])

            # For output projections, slice columns instead of rows because heads compile horizontally
            new_out_proj_weight.append(self.out_proj.weight[:, start_idx:end_idx])

        # Concatenate structural segments back together to synthesize leaner parameter matrices
        new_q_weight = torch.cat(new_q_weight).detach()
        new_k_weight = torch.cat(new_k_weight).detach()
        new_v_weight = torch.cat(new_v_weight).detach()
        new_out_proj_weight = torch.cat(new_out_proj_weight, dim=-1).detach()

        # Re-enable backpropagation gradient tracking paths for optimized parameter components
        new_q_weight.requires_grad = True
        new_k_weight.requires_grad = True
        new_v_weight.requires_grad = True
        new_out_proj_weight.requires_grad = True

        new_q_bias = torch.cat(new_q_bias).detach()
        new_q_bias.requires_grad = True
        new_k_bias = torch.cat(new_k_bias).detach()
        new_k_bias.requires_grad = True
        new_v_bias = torch.cat(new_v_bias).detach()
        new_v_bias.requires_grad = True

        # Re-bind modified parameters back into module parameters fields
        self.q_proj.weight = nn.Parameter(new_q_weight)
        self.q_proj.bias = nn.Parameter(new_q_bias)
        self.k_proj.weight = nn.Parameter(new_k_weight)
        self.k_proj.bias = nn.Parameter(new_k_bias)
        self.v_proj.weight = nn.Parameter(new_v_weight)
        self.v_proj.bias = nn.Parameter(new_v_bias)
        self.out_proj.weight = nn.Parameter(new_out_proj_weight)
        # Update dimensional attribute values inside the class object records to accurately match current configurations
        self.num_heads = len(reserve_head_index)
        self.embed_dim = self.head_dim * self.num_heads
        self.q_proj.out_features = self.embed_dim
        self.k_proj.out_features = self.embed_dim
        self.v_proj.out_features = self.embed_dim

    def _set_skip_embed_dim_check(self):
        self.skip_embed_dim_check = True

    def _pad_masks(self, key_padding_mask, attn_mask):
        """Pads masks to match targets when explicit key/value biases are concatenated."""

        # When bias vectors add custom static token tracks onto the sequence len, masks must be scaled out to prevent errors
        if attn_mask is not None:
            shape = attn_mask.size()[:-1] + torch.Size([1])
            attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(shape)], dim=-1)

        if key_padding_mask is not None:
            shape = key_padding_mask.size()[:-1] + torch.Size([1])
            key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_zeros(shape)], dim=-1)

        return key_padding_mask, attn_mask

    def _add_bias(self, k, v, key_padding_mask, attn_mask, bsz):
        """Appends trained positional bias tokens directly onto key and value tracking."""

        assert self.bias_k is not None or self.bias_v is not None
        key_padding_mask, attn_mask = self._pad_masks(key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # Concatenate structural bias frames directly along the historical sequence dimension row bounds
        return torch.cat([k, self.bias_k.repeat(1, bsz, 1)]), torch.cat([v, self.bias_v.repeat(1, bsz, 1)]), key_padding_mask, attn_mask

    def _append_zero_attn(self, k, v, key_padding_mask, attn_mask):
        """Concatenates an explicit padding zero-channel sequence element to key-value variables."""

        # Establish target dimension sizes tracking spatial metrics matching current key states
        zero_attn_shape = k.size()[:-2] + torch.Size([1]) + k.size()[-1:]
        key_padding_mask, attn_mask = self._pad_masks(key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # Concatenate hard zeros directly to sequences; this provides attention vectors a fallback option to distribute weights onto
        return torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=-2), torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=-2), key_padding_mask, attn_mask

    def forward(self, query, key, value, key_padding_mask = None, incremental_state = None, need_weights = True, static_kv = False, attn_mask = None, before_softmax = False, need_head_weights = False):
        # Head tracking configurations parameter adjustment rules check
        if need_head_weights: need_weights = True
        # Unpack input tensor structural dimensions
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len

        # Run shape checks to guarantee operational calculations match up safely
        if not self.skip_embed_dim_check: assert (embed_dim == self.embed_dim)
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        # If localized key inputs are defined, check internal dimension metrics and shapes match values
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert value is not None
                assert src_len, key_bsz == value.shape[:2]

        if (not self.onnx_trace and incremental_state is None and not static_kv and not torch.jit.is_scripting() and not self.skip_embed_dim_check):
            assert key is not None and value is not None

            return F.multi_head_attention_forward(
                query, 
                key, 
                value, 
                self.embed_dim, 
                self.num_heads, 
                torch.empty([0]), 
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)), 
                self.bias_k, 
                self.bias_v, 
                self.add_zero_attn, 
                self.dropout_module.p, 
                self.out_proj.weight, 
                self.out_proj.bias, 
                self.training or self.dropout_module.apply_during_inference, 
                key_padding_mask.bool() if key_padding_mask is not None else None, 
                need_weights, 
                attn_mask, 
                use_separate_proj_weight=True, 
                q_proj_weight=self.q_proj.weight, 
                k_proj_weight=self.k_proj.weight, 
                v_proj_weight=self.v_proj.weight
            )

        if incremental_state is not None:
            # Extract step context out of the persistent data objects dictionary
            saved_state = self._get_input_buffer(incremental_state)

            if saved_state is not None and "prev_key" in saved_state:
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else: saved_state = None

        # Execute structural forward matrix projections based on current operational attention styles
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                # Handle tracking adjustments when search beams span width dimensions completely
                if self.beam_size > 1 and bsz == key.size(1):
                    key = key.view(key.size(0), -1, self.beam_size, key.size(2))[:, :, 0, :]
                    if key_padding_mask is not None: key_padding_mask = key_padding_mask.view(-1, self.beam_size, key_padding_mask.size(1))[:, 0, :]

                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        # Apply spatial scaling corrections directly onto query distributions
        q *= self.scaling
        if self.bias_k is not None:
            assert self.bias_v is not None
            k, v, attn_mask, key_padding_mask = self._add_bias(k, v, attn_mask, key_padding_mask, bsz)

        # Reshape data structures to run simultaneous batch parallel computations on individual heads:
        q = (q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1))
        kv_bsz = bsz 

        if k is not None:
            kv_bsz = k.size(1)
            k = (k.contiguous().view(-1, kv_bsz * self.num_heads, self.head_dim).transpose(0, 1))

        if v is not None: v = (v.contiguous().view(-1, kv_bsz * self.num_heads, self.head_dim).transpose(0, 1))
        # Merge historical tensor into current tracking variables if sequential logs are found
        if saved_state is not None:
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None

                kv_bsz = _prev_key.size(0)
                # Reshape tracked matrix patterns to align with operational shapes
                prev_key = _prev_key.view(kv_bsz * self.num_heads, -1, self.head_dim)

                if static_kv: k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)

                src_len = k.size(1)

            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None or kv_bsz == _prev_value.size(0)
                prev_value = _prev_value.view(kv_bsz * self.num_heads, -1, self.head_dim)
                if static_kv: v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)

            prev_key_padding_mask = None
            if "prev_key_padding_mask" in saved_state: prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None

            # Append padding tracking masks safely
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask, 
                prev_key_padding_mask=prev_key_padding_mask, 
                batch_size=kv_bsz, 
                src_len=k.size(1), 
                static_kv=static_kv
            )

            # Store updated sequence updates back inside memory storage for subsequent step tracking iterations
            saved_state["prev_key"] = k.view(kv_bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(kv_bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask

            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        assert k is not None
        assert k.size(1) == src_len

        if key_padding_mask is not None and key_padding_mask.dim() == 0: key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == kv_bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k, v, key_padding_mask, attn_mask = self._append_zero_attn(k=k, v=v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        if self.encoder_decoder_attention and bsz != kv_bsz:
            # Deploy customized Einstein summation configurations to bridge dimension gaps safely if batch dimensions vary
            attn_weights = torch.einsum(
                "bxhtd,bhsd->bxhts", 
                q.view((kv_bsz, -1, self.num_heads) + q.size()[1:]), 
                k.view((kv_bsz, self.num_heads) + k.size()[1:])
            )

            attn_weights = attn_weights.reshape((-1,) + attn_weights.size()[-2:])
        else: attn_weights = q.bmm(k.transpose(1, 2)) # Standard batch matrix multiplication across matching rank

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # Apply structural masks by injecting hard negative infinity to suppress selected attention pathways completely
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace: attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(kv_bsz, -1, self.num_heads, tgt_len, src_len)
            # Replace active padding tracking paths with extreme negative floor limits to nullify activation probability scores
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(torch.bool), float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # Break out early if calculations call for raw pre-softmax values
        if before_softmax: return attn_weights, v

        # Normalize attention scores into standard valid probability distributions
        attn_weights_float = softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        # Apply structured drop out regularizations directly over probability maps
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = None
        # Gather context feature representations by mapping attention probabilities against value frames
        if self.encoder_decoder_attention and bsz != kv_bsz:
            attn = torch.einsum(
                "bxhts,bhsd->bxhtd", 
                attn_probs.view((kv_bsz, -1, self.num_heads) + attn_probs.size()[1:]), 
                v.view((kv_bsz, self.num_heads) + v.size()[1:])
            )

            attn = attn.reshape((-1,) + attn.size()[-2:])
        else: attn = attn_probs.bmm(v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        # Transpose values back to standard formats: [Batch * Heads, Length, HeadDim] -> [Length, Batch, EmbedDim]
        attn = attn.contiguous().view(tgt_len, bsz, self.embed_dim) if self.onnx_trace and attn.size(1) == 1 else attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        # Route through final linear output mapping layer
        attn = self.out_proj(attn)
        attn_weights = None

        # If requested, format attention weight summary matrices to return alongside context
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights: 
                # Average scores across individual head channels to synthesize global visualization charts
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(key_padding_mask, prev_key_padding_mask, batch_size, src_len, static_kv):
        """Merges historical step padding specifications with incoming step structural mask lists."""

        if prev_key_padding_mask is not None and static_kv: 
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None: 
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), key_padding_mask.float()], dim=1)
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros((batch_size, src_len - prev_key_padding_mask.size(1)), device=prev_key_padding_mask.device)
                new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), filler.float()], dim=1)
            else: 
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros((batch_size, src_len - key_padding_mask.size(1)), device=key_padding_mask.device)
                new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
            else: 
                new_key_padding_mask = key_padding_mask.float()
        else: 
            new_key_padding_mask = prev_key_padding_mask

        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(self, incremental_state, new_order):
        """Rearranges memory caching channels relative to active branch shifts during candidate generation."""

        input_buffer = self._get_input_buffer(incremental_state)

        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]

                if input_buffer_k is not None:
                    if self.encoder_decoder_attention:
                        if input_buffer_k.size(0) * self.beam_size == new_order.size(0): 
                            return incremental_state
                        elif self.beam_size > 1: 
                            # Slice out specific entries matching active operational routes chosen by search logs
                            input_buffer[k] = input_buffer_k.index_select(0, new_order.reshape(-1, self.beam_size)[:, 0] // self.beam_size)
                        else: 
                            input_buffer[k] = input_buffer_k.index_select(0, new_order)
                    else: 
                        input_buffer[k] = input_buffer_k.index_select(0, new_order)

            # Re-save organized buffer modifications back down into step execution environment histories
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)

        return incremental_state

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size

    def _get_input_buffer(self, incremental_state):
        result = self.get_incremental_state(incremental_state, "attn_state")
        return result if result is not None else {}

    def _set_input_buffer(self, incremental_state, buffer):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def upgrade_state_dict_named(self, state_dict, name):
        """Splits unified `in_proj_weight` vectors into distinct Q, K, V sub-projection weights."""

        prefix = name + "." if name != "" else ""
        items_to_add, keys_to_remove = {}, []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                dim = int(state_dict[k].shape[0] / 3)
                # Segment unified tensors into three isolated linear parameters maps
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)
                k_bias = prefix + "in_proj_bias"
                # Check if bias parameters follow identical formatting layouts and process them if present
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][dim : 2 * dim]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]
                    keys_to_remove.append(prefix + "in_proj_bias")

        # Delete deprecated outdated state key references out of the configuration map dictionary
        for k in keys_to_remove:
            del state_dict[k]

        # Inject newly structured parameter updates into the active runtime initialization configurations list
        for key, value in items_to_add.items():
            state_dict[key] = value

def init_bert_params(module):
    """Initializes sub-module layers using a BERT normal distribution setup."""

    def normal_(data):
        # In-place copy normal values sampled at mean 0.0 and standard deviation 0.02
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    # Apply specialized initialization rules across various structural layer components
    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None: module.bias.data.zero_()

    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None: module.weight.data[module.padding_idx].zero_()

    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)

def make_conv_pos(e, k, g):
    """Creates a 1D convolutional positional embedding sequence layer with weight normalization applied."""

    pos_conv = nn.Conv1d(e, e, kernel_size=k, padding=k // 2, groups=g)
    dropout = 0
    # Initialize weights normally leveraging mathematical equations matching specialized feature scales
    nn.init.normal_(pos_conv.weight, mean=0, std=math.sqrt((4 * (1.0 - dropout)) / (k * e)))
    nn.init.constant_(pos_conv.bias, 0)
    # Compile layers within Sequential wrapping structures, injecting Weight Normalization parametrizations
    return nn.Sequential(nn.utils.parametrizations.weight_norm(pos_conv, name="weight", dim=2), SamePad(k), nn.GELU())

def index_put(tensor, indices, value):
    """Performs an in-place tensor modification by putting values into specified indices."""

    # Write incoming value tracking fields into selected locations matching index parameters
    tensor[indices] = value
    return tensor

def pad_to_multiple(x, multiple, dim=-1, value=0):
    """
    Pads a tensor on a given dimension to make its size a multiple of a specified value.

    Args:
        x (torch.Tensor or None): The input tensor to pad.
        multiple (int): The target multiple value for padding.
        dim (int, optional): The dimension along which to pad. Defaults to -1.
        value (float, optional): The padding value. Defaults to 0.

    Returns:
        tuple: (padded_tensor, remainder)
            - padded_tensor (torch.Tensor or None): The padded tensor or None if input is None.
            - remainder (int): The number of padded elements added.
    """

    if x is None: return None, 0

    tsz = x.size(dim)
    m = tsz / multiple
    # Calculate how many elements are missing to reach the next multiple
    remainder = math.ceil(m) * multiple - tsz
    # If already a multiple, return the original tensor
    if m.is_integer(): return x, 0
    # Create the correct padding tuple based on the target dimension
    return F.pad(x, (*((0,) * (-1 - dim) * 2), 0, remainder), value=value), remainder

def prune_state_dict(state_dict, model_cfg):
    """
    Prunes layers from a state dictionary based on configuration constraints.

    Args:
        state_dict (dict): The original state dictionary containing model weights.
        model_cfg (object): The model configuration object (can be a DictConfig or standard object).

    Returns:
        dict: The pruned state dictionary with mapped layer indices.
    """

    arch = None
    if model_cfg is not None: arch = (model_cfg._name if isinstance(model_cfg, DictConfig) else getattr(model_cfg, "arch", None))
    # Return early if configuration is empty or architecture does not require pruning
    if not model_cfg or arch is None or arch == "ptt_transformer": return state_dict
    encoder_layers_to_keep = getattr(model_cfg, "encoder_layers_to_keep", None)
    decoder_layers_to_keep = getattr(model_cfg, "decoder_layers_to_keep", None)
    if not encoder_layers_to_keep and not decoder_layers_to_keep: return state_dict

    def create_pruning_pass(layers_to_keep, layer_name):
        # Parse comma-separated string of layer indices to keep and sort them
        keep_layers = sorted(int(layer_string) for layer_string in layers_to_keep.split(","))
        mapping_dict = {}
        # Create a mapping from old indices to new contiguous indices
        for i in range(len(keep_layers)):
            mapping_dict[str(keep_layers[i])] = str(i)

        return {"substitution_regex": re.compile(r"^{layer}.*\.layers\.(\d+)".format(layer=layer_name)), "mapping_dict": mapping_dict}

    pruning_passes, new_state_dict = [], {}
    if encoder_layers_to_keep: pruning_passes.append(create_pruning_pass(encoder_layers_to_keep, "encoder"))
    if decoder_layers_to_keep: pruning_passes.append(create_pruning_pass(decoder_layers_to_keep, "decoder"))

    for layer_name in state_dict.keys():
        match = re.search(r"\.layers\.(\d+)\.", layer_name)
        # If the parameter does not belong to a numbered layer, keep it as is
        if not match:
            new_state_dict[layer_name] = state_dict[layer_name]
            continue

        original_layer_number = match.group(1)
        for pruning_pass in pruning_passes:
            # Check if this layer is marked to be kept and matches the current pass's regex
            if original_layer_number in pruning_pass["mapping_dict"] and pruning_pass["substitution_regex"].search(layer_name):
                substitution_match = pruning_pass["substitution_regex"].search(layer_name)
                # Reconstruct the layer name with the new pruned index mapping
                new_state_dict[
                    (
                        layer_name[: substitution_match.start(1)] + 
                        pruning_pass["mapping_dict"][original_layer_number] + 
                        layer_name[substitution_match.end(1) :]
                    )
                ] = state_dict[layer_name]

    # Clean up the configuration properties after pruning is done
    with open_dict(model_cfg) if isinstance(model_cfg, DictConfig) else contextlib.ExitStack():
        if hasattr(model_cfg, "encoder_layers_to_keep"): model_cfg.encoder_layers_to_keep = None
        if hasattr(model_cfg, "decoder_layers_to_keep"): model_cfg.decoder_layers_to_keep = None

    return new_state_dict

def get_activation_fn(activation):
    """
    Factory function to retrieve an activation function callable based on a string identifier.

    Args:
        activation (str): The name of the activation function.

    Returns:
        callable: The requested activation function or module reference.

    Raises:
        ValueError: If an unsupported activation function name is requested.
    """

    def relu_squared(x):
        return F.relu(x).pow(2)

    def gelu(x):
        # Internal gelu casting safely handles fp16 precision stability
        return nn.functional.gelu(x.float()).type_as(x)
    
    def gelu_accurate(x):
        if not hasattr(gelu_accurate, "_a"):
            gelu_accurate._a = math.sqrt(2 / math.pi)
        # Approximation formula for accurate/fast GeLU computation
        return (0.5 * x * (1 + (gelu_accurate._a * (x + 0.044715 * x.pow(3))).tanh()))

    if activation == "relu": return F.relu
    elif activation == "relu_squared": return relu_squared
    elif activation == "gelu": return gelu
    elif activation == "gelu_fast": return gelu_accurate
    elif activation == "gelu_accurate": return gelu_accurate
    elif activation == "tanh": return torch.tanh
    elif activation == "linear": return lambda x: x
    elif activation == "swish": return nn.SiLU
    else: raise ValueError(f"Activation function '{activation}' is not supported.")

class SamePad(nn.Module):
    """
    Applies right-side cropping to match standard 'SAME' padding behavior in convolutions.
    """

    def __init__(
        self, 
        kernel_size, 
        causal=False
    ):
        super().__init__()
        if causal: self.remove = kernel_size - 1
        else: self.remove = int(kernel_size % 2 == 0)

    def forward(self, x):
        if self.remove > 0: x = x[:, :, : -self.remove]
        return x

class TransformerSentenceEncoderLayer(nn.Module):
    """
    Standard Transformer Encoder Layer supporting post-LN and pre-LN styles.
    """

    def __init__(
        self, 
        embedding_dim = 768, 
        ffn_embedding_dim = 3072, 
        num_attention_heads = 8, 
        dropout = 0.1, 
        attention_dropout = 0.1, 
        activation_dropout = 0.1, 
        activation_fn = "relu", 
        layer_norm_first = False
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = get_activation_fn(activation_fn)
        # Self-attention submodule
        self.self_attn = MultiheadAttention(self.embedding_dim, num_attention_heads, dropout=attention_dropout, self_attention=True)
        # Regularization dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)
        # Normalization layers
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim, eps=1e-5, elementwise_affine=True)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim, eps=1e-5, elementwise_affine=True)
        # Bind the correct forward routing style dynamically
        self.forward = self.forward_layer_norm_first if layer_norm_first else self.forward_non_layer_norm_first
    
    def forward_layer_norm_first(self, x, self_attn_mask=None, self_attn_padding_mask=None, need_weights=False, att_args=None):
        residual = x

        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask, attn_mask=self_attn_mask, need_weights=False)
        x = residual + self.dropout1(x)

        residual = x
        x = self.fc2(self.dropout2(self.activation_fn(self.fc1(self.final_layer_norm(x)))))

        layer_result = x
        x = residual + self.dropout3(x)

        return x, (attn, layer_result)

    def forward_non_layer_norm_first(self, x, self_attn_mask=None, self_attn_padding_mask=None, need_weights=False, att_args=None):
        residual = x

        x, attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask, need_weights=False)
        x = self.self_attn_layer_norm(residual + self.dropout1(x))

        residual = x
        x = self.fc2(self.dropout2(self.activation_fn(self.fc1(x))))

        layer_result = x
        x = self.final_layer_norm(residual + self.dropout3(x))

        return x, (attn, layer_result)

class AdapterFast(nn.Module):
    """
    Fast Parallel/Batched Multi-head Adapter Module for Parameter Efficient Fine Tuning.
    """

    def __init__(
        self, 
        adapter_num, 
        input_dim, 
        hidden_dim, 
        act_fn
    ):
        super().__init__()
        self.adapter_num = adapter_num
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Batched parameters for processing unique adapter weights in a single tensor operation
        self.W_a = nn.Parameter(torch.empty(adapter_num, hidden_dim, input_dim))
        self.W_b = nn.Parameter(torch.empty(adapter_num, input_dim, hidden_dim))
        self.b_a = nn.Parameter(torch.empty(adapter_num, hidden_dim))
        self.b_b = nn.Parameter(torch.empty(adapter_num, input_dim))
        self.ln_W = nn.Parameter(torch.empty(adapter_num, input_dim))
        self.ln_b = nn.Parameter(torch.empty(adapter_num, input_dim))
        self.act_fn = nn.Identity()
        if act_fn == "relu": self.act_fn = nn.ReLU()
        elif act_fn == "gelu": self.act_fn = nn.GELU()
        elif act_fn == "selu": self.act_fn = nn.SELU()
        else: raise ValueError(f"Activation function '{act_fn}' is not supported in AdapterFast module.")
        self.input_dim = input_dim
        self.reset_parameters()

    def reset_parameters(self):
        # Uniform Kaiming Initialization across batch components
        for ii in range(self.adapter_num):
            nn.init.kaiming_uniform_(self.W_a[ii], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.W_b[ii], a=math.sqrt(5))
            # Bound handling for consistent linear initialization
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_a[ii])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b_a[ii], -bound, bound)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_b[ii])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b_b[ii], -bound, bound)
        
        nn.init.ones_(self.ln_W)
        nn.init.zeros_(self.ln_b)

    def forward(self, x, adapter_id):
        ii = adapter_id
        # Evaluates specialized LayerNorm and Linear transforms tied to adapter index `ii`
        return F.linear(
            self.act_fn(
                F.linear(
                    F.layer_norm(
                        x, 
                        (self.input_dim, ), 
                        self.ln_W[ii], 
                        self.ln_b[ii]
                    ), 
                    self.W_a[ii], 
                    self.b_a[ii]
                )
            ), 
            self.W_b[ii], 
            self.b_b[ii]
        )
    
    def extra_repr(self):
        return (
            'adapter={}, input_dim={}, hidden_dim={}'.format(self.adapter_num, self.input_dim, self.hidden_dim)
        )

class FeedForwardModule(nn.Module):
    """
    Conformer Feed Forward module processing sequential inputs.
    """

    def __init__(
        self, 
        input_feat, 
        hidden_units, 
        dropout1, 
        dropout2, 
        activation_fn="swish", 
        bias=True
    ):
        super(FeedForwardModule, self).__init__()
        self.layer_norm = nn.LayerNorm(input_feat, eps=1e-5, elementwise_affine=True)
        self.w_1 = nn.Linear(input_feat, hidden_units, bias=bias)
        self.w_2 = nn.Linear(hidden_units, input_feat, bias=bias)
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        # Instantiate the functional/callable wrapper via the helper factory
        self.activation = get_activation_fn(activation_fn)(hidden_units)

    def forward(self, x):
        return self.dropout2(
            self.w_2(
                self.dropout1(
                    self.activation(
                        self.w_1(
                            self.layer_norm(x)
                        )
                    )
                )
            )
        )

class ConvolutionModule(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        channels, 
        depthwise_kernel_size, 
        dropout, 
        activation_fn="swish", 
        bias=False
    ):
        super(ConvolutionModule, self).__init__()
        # Enforce symmetrical padding compatibility requirements
        assert (depthwise_kernel_size - 1) % 2 == 0
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True)
        # Pointwise convolution expands the channel dimension for Gated Linear Unit processing
        self.pointwise_conv1 = nn.Conv1d(embed_dim, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(channels, channels, depthwise_kernel_size, stride=1, padding=(depthwise_kernel_size - 1) // 2, groups=channels, bias=bias)
        self.batch_norm = nn.BatchNorm1d(channels)
        self.activation = get_activation_fn(activation_fn)(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, embed_dim, kernel_size=1, stride=1, padding=0, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Rearrange dimensionality format to adapt to PyTorch's native 1D Conv structure
        return self.dropout(self.pointwise_conv2(self.activation(self.batch_norm(self.depthwise_conv(self.glu(self.pointwise_conv1(self.layer_norm(x).transpose(1, 2)))))))).transpose(1, 2)

def rotate_half(x):
    """
    Splits the final embedding dimension into halves and rotates their coordinates.
    """

    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]

    return torch.cat((-x2, x1), dim=x1.ndim - 1)

def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    """
    Applies Rotary Position Embedding (RoPE) matrices to query and key tensors.
    """

    cos, sin = (cos[offset : q.shape[0] + offset, ...], sin[offset : q.shape[0] + offset, ...])

    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RotaryPositionalEmbedding(nn.Module):
    """
    Computes sin/cos dynamic frequency caches for RoPE positional embeddings.
    """

    def __init__(
        self, 
        dim, 
        base=10000, 
        precision=torch.half
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = 0
        self.cos_cached = torch.empty(self.seq_len_cached, 1, 1, dim)
        self.sin_cached = torch.empty(self.seq_len_cached, 1, 1, dim)
        self.precision = precision

    def forward(self, x, seq_len = 0):
        # Regenerate caches when the maximum execution sequence boundary expands
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len

            freqs = torch.einsum("i,j->ij", torch.arange(seq_len, device=x.device).type_as(self.inv_freq), self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self.cos_cached = emb.cos().view(emb.size(0), 1, 1, emb.size(1))
            self.sin_cached = emb.sin().view(emb.size(0), 1, 1, emb.size(1))

        return self.cos_cached, self.sin_cached

class ESPNETMultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention component following ESPNET structural standards.
    """

    def __init__(
        self, 
        n_feat, 
        n_head, 
        dropout
    ):
        super(ESPNETMultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward_qkv(self, query, key, value, **kwargs):
        n_batch = query.size(0)
        # Returns mapped projections formatted as (Batch, Head, SeqLen, HeadDim)
        return (
            self.linear_q(query).view(n_batch, -1, self.h, self.d_k).transpose(1, 2), 
            self.linear_k(key).view(n_batch, -1, self.h, self.d_k).transpose(1, 2), 
            self.linear_v(value).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        )

    def forward_attention(self, value, scores, mask):
        n_batch = value.size(0)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2).to(bool), float("-inf"))

        self.attn = scores.softmax(dim=-1)
        return self.linear_out(((self.dropout(self.attn) @ value).transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k))) 

    def forward(self, query, key, value, key_padding_mask=None, **kwargs):
        # Transpose input shapes from (SeqLen, Batch, Dim) to (Batch, SeqLen, Dim)
        q, k, v = self.forward_qkv(query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1))
        return self.forward_attention(v, (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k), key_padding_mask).transpose(0, 1), None

class RelPositionMultiHeadedAttention(ESPNETMultiHeadedAttention):
    """
    Multi-Head Attention featuring Relative Positional Encoding structures.
    """

    def __init__(self, n_feat, n_head, dropout):
        super().__init__(n_feat, n_head, dropout)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        # Shifts attention matrix results cleanly to represent accurate relative index offsets
        x = torch.cat(
            [
                torch.zeros(
                    (*x.size()[:3], 1), 
                    device=x.device, 
                    dtype=x.dtype
                ), 
                x
            ], 
            dim=-1
        ).view(*x.size()[:2], x.size(3) + 1, x.size(2))[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]

        return x

    def forward(self, query, key, value, pos_emb, key_padding_mask=None, **kwargs):
        pos_emb = pos_emb.transpose(0, 1)
        q, k, v = self.forward_qkv(query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1))
        q = q.transpose(1, 2)

        return self.forward_attention(
            v, (
                (
                    (q + self.pos_bias_u).transpose(1, 2) @ k.transpose(-2, -1)
                ) + self.rel_shift(
                    ((q + self.pos_bias_v).transpose(1, 2) @ self.linear_pos(pos_emb).view(pos_emb.size(0), -1, self.h, self.d_k).transpose(1, 2).transpose(-2, -1))
                )
        ) / math.sqrt(self.d_k), key_padding_mask).transpose(0, 1), None

class RotaryPositionMultiHeadedAttention(ESPNETMultiHeadedAttention):
    """
    Multi-Head Attention integrating Rotary Positional Embeddings.
    """

    def __init__(
        self, 
        n_feat, 
        n_head, 
        dropout, 
        precision, 
        rotary_emd_base=10000
    ):
        super().__init__(n_feat, n_head, dropout)
        precision = torch.float
        self.rotary_ndims = self.d_k
        if precision == "fp16": precision = torch.half
        self.rotary_emb = RotaryPositionalEmbedding(self.rotary_ndims, base=rotary_emd_base, precision=precision)

    def forward(self, query, key, value, key_padding_mask=None, **kwargs):
        T, B, _ = value.size()

        query = query.view(T, B, self.h, self.d_k)
        key = key.view(T, B, self.h, self.d_k)

        value = value.view(T, B, self.h, self.d_k)
        cos, sin = self.rotary_emb(value, seq_len=T)

        query, key = apply_rotary_pos_emb(query, key, cos, sin, offset=0)
        query = query.view(T, B, self.h * self.d_k)

        key = key.view(T, B, self.h * self.d_k)
        value = value.view(T, B, self.h * self.d_k)
        q, k, v = self.forward_qkv(query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1))

        return self.forward_attention(v, (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k), key_padding_mask).transpose(0, 1), None

class ConformerEncoderLayer(nn.Module):
    """
    Conformer block structure featuring interleaved feed-forward modules.
    """

    def __init__(self, embed_dim, ffn_embed_dim, attention_heads, dropout, use_fp16, depthwise_conv_kernel_size=31, activation_fn="swish", attn_type=None, pos_enc_type="abs"):
        self.pos_enc_type = pos_enc_type
        super(ConformerEncoderLayer, self).__init__()
        # Macaron-style sandwich feed forward networks
        self.ffn1 = FeedForwardModule(embed_dim, ffn_embed_dim, dropout, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True)
        self.self_attn_dropout = nn.Dropout(dropout)
        # Configure self-attention mechanics matching requested types
        if attn_type == "espnet":
            if self.pos_enc_type == "rel_pos": self.self_attn = RelPositionMultiHeadedAttention(embed_dim, attention_heads, dropout=dropout)
            elif self.pos_enc_type == "rope": self.self_attn = RotaryPositionMultiHeadedAttention(embed_dim, attention_heads, dropout=dropout, precision=use_fp16)
            elif self.pos_enc_type == "abs": self.self_attn = ESPNETMultiHeadedAttention(embed_dim, attention_heads, dropout=dropout)
            else: raise ValueError(f"Unsupported positional encoding type '{self.pos_enc_type}' for espnet attention module.")
        else: self.self_attn = MultiheadAttention(embed_dim, attention_heads, dropout=dropout)
        self.conv_module = ConvolutionModule(embed_dim=embed_dim, channels=embed_dim, depthwise_kernel_size=depthwise_conv_kernel_size, dropout=dropout, activation_fn=activation_fn)
        self.ffn2 = FeedForwardModule(embed_dim, ffn_embed_dim, dropout, dropout, activation_fn=activation_fn)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x, encoder_padding_mask, position_emb = None):
        residual = x
        # 0.5 scaling factor matches standard Conformer architecture design
        x = self.ffn1(x) * 0.5 + residual
        residual = x

        x = self.self_attn_layer_norm(x)
        if self.pos_enc_type == "rel_pos": x, attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask, pos_emb=position_emb, need_weights=False)
        else: x, attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask, need_weights=False)

        x = self.self_attn_dropout(x)
        x = x + residual
        residual = x

        # Convolution processing pass (requires dimension swapping to adapt sequence orders)
        x = residual + self.conv_module(x.transpose(0, 1)).transpose(0, 1)
        residual = x
        x = self.ffn2(x)

        layer_result = x
        x = self.final_layer_norm(x * 0.5 + residual)

        return x, (attn, layer_result)

class ConformerWav2Vec2EncoderLayer(ConformerEncoderLayer):
    def forward(
        self, 
        x, 
        self_attn_mask=None, 
        self_attn_padding_mask=None, 
        need_weights=False, 
        att_args=None, 
        position_emb=None
    ):
        """
        Subclass wrapper adjusting ConformerEncoderLayer interface format for compatibility with Wav2Vec2 pipelines.
        """

        return super().forward(
            x, 
            self_attn_padding_mask, 
            position_emb
        )

class TransformerSentenceEncoderWithAdapterLayer(TransformerSentenceEncoderLayer):
    """
    Transformer Sentence Encoder Layer extended with a fast multi-head Adapter variant.
    """

    def __init__(
        self, 
        embedding_dim = 768, 
        ffn_embedding_dim = 3072, 
        num_attention_heads = 8, 
        dropout = 0.1, 
        attention_dropout = 0.1, 
        activation_dropout = 0.1, 
        activation_fn = "relu", 
        layer_norm_first = False, 
        adapter_num=201, 
        adapter_dim=64, 
        adapter_act_fn="relu"
    ):
        super().__init__(
            embedding_dim=embedding_dim, 
            ffn_embedding_dim=ffn_embedding_dim, 
            num_attention_heads=num_attention_heads, 
            dropout=dropout, 
            attention_dropout=attention_dropout, 
            activation_dropout=activation_dropout, 
            activation_fn=activation_fn, 
            layer_norm_first=layer_norm_first
        )
        self.adapter_num = adapter_num
        self.adapter_dim = adapter_dim
        # Initialize parallel multi-component Adapter instance
        self.adapter_layer = AdapterFast(adapter_num, self.embedding_dim, self.adapter_dim, adapter_act_fn)

    def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None, need_weights=False, att_args=None, corpus_key=None):
        # Delegate core computations to the standard transformer structure base class
        x, (attn, layer_result) = super().forward(x=x, self_attn_mask=self_attn_mask, self_attn_padding_mask=self_attn_padding_mask, need_weights=need_weights, att_args=att_args)
        # Guard rails verifying presence and single-domain homogeneity of adapter indexing arrays
        assert corpus_key is not None
        assert len(set(corpus_key)) == 1

        # Add residual-like pathway scaling through the selected target domain adapter index
        return x + self.adapter_layer(x, corpus_key[0]), (attn, layer_result)

class TransposeLast(nn.Module):
    """
    Utility module to selectively deconstruct input containers and flip the final dimensions.
    """

    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        # Extract specific elements if input is a structured list/tuple container
        if self.deconstruct_idx is not None: x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)

class TransformerEncoder(nn.Module):
    """
    Flexible Backing Encoder Layer Router supporting standard Transformer, Conformer, or Adapter variations.
    """

    def build_encoder_layer(self, args, **kwargs):
        if args.layer_type == "transformer": 
            layer = TransformerSentenceEncoderLayer(
                embedding_dim=self.embedding_dim, 
                ffn_embedding_dim=args.encoder_ffn_embed_dim, 
                num_attention_heads=args.encoder_attention_heads, 
                dropout=self.dropout, 
                attention_dropout=args.attention_dropout, 
                activation_dropout=args.activation_dropout, 
                activation_fn=args.activation_fn, 
                layer_norm_first=args.layer_norm_first
            )
        elif args.layer_type == "conformer": 
            layer = ConformerWav2Vec2EncoderLayer(
                embed_dim=self.embedding_dim, 
                ffn_embed_dim=args.encoder_ffn_embed_dim, 
                attention_heads=args.encoder_attention_heads, 
                dropout=args.dropout, 
                depthwise_conv_kernel_size=args.depthwise_conv_kernel_size, 
                activation_fn="swish", 
                attn_type=args.attn_type, 
                use_fp16=args.fp16, 
                pos_enc_type="abs"
            )
        elif args.layer_type == "trf_adp":
            use_adp = False
            # Decide whether to mount Adapter units uniformly or selectively by sliced ranges
            if args.adp_trf_idx == "all": use_adp = True
            else: 
                if kwargs.get("layer_idx", None) in list(range(*[int(g) for g in args.adp_trf_idx.split(":")])): use_adp = True

            layer = (
                TransformerSentenceEncoderWithAdapterLayer(
                    embedding_dim=self.embedding_dim, 
                    ffn_embedding_dim=args.encoder_ffn_embed_dim, 
                    num_attention_heads=args.encoder_attention_heads, 
                    dropout=self.dropout, 
                    attention_dropout=args.attention_dropout, 
                    activation_dropout=args.activation_dropout, 
                    activation_fn=args.activation_fn, 
                    layer_norm_first=args.layer_norm_first, 
                    adapter_num=args.adp_num, 
                    adapter_dim=args.adp_dim, 
                    adapter_act_fn=args.adp_act_fn
                )
            ) if use_adp else (
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim, 
                    ffn_embedding_dim=args.encoder_ffn_embed_dim, 
                    num_attention_heads=args.encoder_attention_heads, 
                    dropout=self.dropout, 
                    attention_dropout=args.attention_dropout, 
                    activation_dropout=args.activation_dropout, 
                    activation_fn=args.activation_fn, 
                    layer_norm_first=args.layer_norm_first
                )
            )

        return layer

    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.required_seq_len_multiple = args.required_seq_len_multiple
        pos_conv_depth = getattr(args, "pos_conv_depth", 1)
        if pos_conv_depth > 1:
            num_layers = args.pos_conv_depth
            k = max(3, args.conv_pos // num_layers)

            def make_conv_block(e, k, g, l):
                # Generates repetitive blocks mapping sequence dimensions for deep position convolutions
                return nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.Conv1d(
                                e, 
                                e, 
                                kernel_size=k, 
                                padding=k // 2, 
                                groups=g
                            ), 
                            SamePad(k), 
                            TransposeLast(), 
                            nn.LayerNorm(e, eps=1e-5, elementwise_affine=False), 
                            TransposeLast(), 
                            nn.GELU()
                        ) 
                        for _ in range(l)
                    ]
                )

            self.pos_conv = make_conv_block(
                self.embedding_dim, 
                k, 
                args.conv_pos_groups, 
                num_layers
            )
        else: 
            self.pos_conv = make_conv_pos(
                self.embedding_dim, 
                args.conv_pos, 
                args.conv_pos_groups
            )

        # Dynamic construction loop filling sub-layer lists
        self.layers = nn.ModuleList([self.build_encoder_layer(args, layer_idx=ii) for ii in range(args.encoder_layers)])
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = nn.LayerNorm(self.embedding_dim, eps=1e-5, elementwise_affine=True)
        self.layerdrop = args.encoder_layerdrop
        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, layer)
        if self.layer_norm_first and layer is None: x = self.layer_norm(x)
        return x, layer_results

    def extract_features(self, x, padding_mask=None, tgt_layer=None):
        # Mask out padded positions explicitly to clean the feature stream
        x = index_put(x, padding_mask, 0)
        # Apply 1D positional convolution (shape transpositions needed for dimension compatibility)
        x += self.pos_conv(x.transpose(1, 2)).transpose(1, 2)

        if not self.layer_norm_first: x = self.layer_norm(x)

        # Force structural sequence length adjustments matching target step multiples
        x, pad_length = pad_to_multiple(x, self.required_seq_len_multiple, dim=-2, value=0)
        padding_mask, _ = pad_to_multiple(padding_mask, self.required_seq_len_multiple, dim=-1, value=True)

        x = F.dropout(x, p=self.dropout, training=self.training).transpose(0, 1)
        layer_results = []
        r = None

        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random() if self.layerdrop > 0 else 1
            # LayerDrop implementation: drop random layers entirely during model training phases
            if not self.training or (dropout_probability > self.layerdrop):
                x, (z, lr) = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                if i >= 0: layer_results.append((x, z, lr))

            if i == tgt_layer:
                r = x
                break

        if r is not None: x = r
        x = x.transpose(0, 1)
        # Reverse structural adjustments, slicing off dummy tokens used during step alignments
        if pad_length > 0:
            x = x[:, :-pad_length]

            def undo_pad(a, b, c):
                return (a[:-pad_length], b[:-pad_length] if b is not None else b, c[:-pad_length])

            layer_results = [undo_pad(*u) for u in layer_results]

        return x, layer_results

    def max_positions(self):
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict

class ConvFeatureExtractionModel(nn.Module):
    """
    Strided 1D Convolution Network mapping audio waveforms to hidden feature streams.
    """

    def __init__(
        self, 
        conv_layers, 
        dropout = 0.0, 
        mode = "default", 
        conv_bias = False
    ):
        super().__init__()
        assert mode in {"default", "layer_norm"}

        def block(n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (is_layer_norm and is_group_norm) == False

            if is_layer_norm: 
                return nn.Sequential(
                    make_conv(), 
                    nn.Dropout(p=dropout), 
                    nn.Sequential(
                        TransposeLast(), 
                        Fp32LayerNorm(
                            dim, 
                            elementwise_affine=True
                        ), 
                        TransposeLast()
                    ), 
                    nn.GELU()
                )
            elif is_group_norm: 
                return nn.Sequential(
                    make_conv(), 
                    nn.Dropout(p=dropout), 
                    Fp32GroupNorm(dim, dim, affine=True), 
                    nn.GELU()
                )
            else: 
                return nn.Sequential(
                    make_conv(), 
                    nn.Dropout(p=dropout), 
                    nn.GELU()
                )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d, 
                    dim, 
                    k, 
                    stride, 
                    is_layer_norm=mode == "layer_norm", 
                    is_group_norm=mode == "default" and i == 0, 
                    conv_bias=conv_bias
                )
            )

            in_d = dim

    def forward(self, x):
        # Unsqueeze creates the distinct target input channel dimension
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)

        return x

class GradMultiply(torch.autograd.Function):
    """
    Autograd Function that rescales passing gradient bounds without altering forward data outputs.
    """

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        # Scale backprop currents seamlessly during reverse topology loops
        return grad * ctx.scale, None

class BaseFairseqModel(nn.Module):
    """
    Abstract Base class offering state dictionary pruning and quick decoding/eval configurations.
    """

    def __init__(self):
        super().__init__()
        self._is_generation_fast = False

    def extract_features(self, *args, **kwargs):
        return self(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, model_cfg = None, args = None):
        self.upgrade_state_dict(state_dict)
        # Execute global state pruning loops targeting structural matches
        new_state_dict = prune_state_dict(state_dict, model_cfg)
        return super().load_state_dict(new_state_dict, strict)

    def upgrade_state_dict(self, state_dict):
        self.upgrade_state_dict_named(state_dict, "")

    def upgrade_state_dict_named(self, state_dict, name):
        assert state_dict is not None

        def do_upgrade(m, prefix):
            if len(prefix) > 0: prefix += "."
            for n, c in m.named_children():
                name = prefix + n
                if hasattr(c, "upgrade_state_dict_named"): c.upgrade_state_dict_named(state_dict, name)
                elif hasattr(c, "upgrade_state_dict"): c.upgrade_state_dict(state_dict)
                do_upgrade(c, name)

        do_upgrade(self, name)

    def make_generation_fast_(self, **kwargs):
        if self._is_generation_fast: return
        self._is_generation_fast = True

        def apply_remove_weight_norm(module):
            try:
                # Remove parameter wraps safely if PyTorch's newer parametrization architecture or weight_norm is found
                if hasattr(module, "parametrizations") and "weight" in module.parametrizations: parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
                else: nn.utils.remove_weight_norm(module)
            except (AttributeError, ValueError):
                return

        self.apply(apply_remove_weight_norm)
        def apply_make_generation_fast_(module, prefix):
            if len(prefix) > 0: prefix += "."

            base_func = BaseFairseqModel.make_generation_fast_
            for n, m in module.named_modules():
                # Trigger custom sub-module quick optimization steps recursively where defined
                if (m != self and hasattr(m, "make_generation_fast_") and m.make_generation_fast_.__func__ is not base_func): m.make_generation_fast_(name=prefix + n, **kwargs)

        apply_make_generation_fast_(self, "")
        self.eval()

class HubertConfig:
    """
    Configuration dataclass storing hyperparameters for HuBERT model initialization,
    feature extraction setup, masking schedules, and optimization variables.
    """

    def __init__(
        self, 
        _name = None, 
        label_rate = 50, 
        encoder_layers_1 = 3, 
        logit_temp_ctr = 0.1, 
        num_negatives = 100, 
        cross_sample_negatives = 0, 
        ctr_layers = [-6],
        crop_seq_to_multiple = 1,
        extractor_mode = "default", 
        encoder_layers = 12, 
        encoder_embed_dim = 768, 
        encoder_ffn_embed_dim = 3072, 
        encoder_attention_heads = 12, 
        activation_fn = "gelu", 
        layer_type = "transformer", 
        dropout = 0.1, 
        attention_dropout = 0.1, 
        activation_dropout = 0.0, 
        encoder_layerdrop = 0.0, 
        dropout_input = 0.0, 
        dropout_features = 0.0, 
        final_dim = 0, 
        untie_final_proj = False, 
        layer_norm_first = False, 
        conv_feature_layers = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2", 
        conv_bias = False, 
        logit_temp = 0.1, 
        target_glu = False, 
        feature_grad_mult = 1.0, 
        mask_length = 10, 
        mask_prob = 0.65, 
        mask_selection = "static", 
        mask_other = 0.0, 
        no_mask_overlap = False, 
        mask_min_space = 1, 
        mask_channel_length = 10, 
        mask_channel_prob = 0.0, 
        mask_channel_selection = "static", 
        mask_channel_other = 0.0, 
        no_mask_channel_overlap = False, 
        mask_channel_min_space = 1, 
        conv_pos = 128, 
        conv_pos_groups = 16, 
        conv_pos_batch_norm = False, 
        latent_temp = (2, 0.5, 0.999995), 
        skip_masked = False, 
        skip_nomask = False, 
        checkpoint_activations = False, 
        required_seq_len_multiple = 2, 
        depthwise_conv_kernel_size = 31, 
        attn_type = "", 
        pos_enc_type = "abs", 
        fp16 = False
    ):
        self._name = _name
        self.label_rate = label_rate
        self.encoder_layers_1 = encoder_layers_1
        self.logit_temp_ctr = logit_temp_ctr
        self.num_negatives = num_negatives
        self.cross_sample_negatives = cross_sample_negatives
        self.ctr_layers = ctr_layers
        self.crop_seq_to_multiple = crop_seq_to_multiple
        self.extractor_mode = extractor_mode
        self.encoder_layers = encoder_layers
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.activation_fn = activation_fn
        self.layer_type = layer_type
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.dropout_input = dropout_input
        self.dropout_features = dropout_features
        self.final_dim = final_dim
        self.untie_final_proj = untie_final_proj
        self.layer_norm_first = layer_norm_first
        self.conv_feature_layers = conv_feature_layers
        self.conv_bias = conv_bias
        self.logit_temp = logit_temp
        self.target_glu = target_glu
        self.feature_grad_mult = feature_grad_mult
        self.mask_length = mask_length
        self.mask_prob = mask_prob
        self.mask_selection = mask_selection
        self.mask_other = mask_other
        self.no_mask_overlap = no_mask_overlap
        self.mask_min_space = mask_min_space
        self.mask_channel_length = mask_channel_length
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_selection = mask_channel_selection
        self.mask_channel_other = mask_channel_other
        self.no_mask_channel_overlap = no_mask_channel_overlap
        self.mask_channel_min_space = mask_channel_min_space
        self.conv_pos = conv_pos
        self.conv_pos_groups = conv_pos_groups
        self.conv_pos_batch_norm = conv_pos_batch_norm
        self.latent_temp = latent_temp
        self.skip_masked = skip_masked
        self.skip_nomask = skip_nomask
        self.checkpoint_activations = checkpoint_activations
        self.required_seq_len_multiple = required_seq_len_multiple
        self.depthwise_conv_kernel_size = depthwise_conv_kernel_size
        self.attn_type = attn_type
        self.pos_enc_type = pos_enc_type
        self.fp16 = fp16

class HubertModel(BaseFairseqModel):
    """
    HuBERT (Hidden-Unit BERT) Model implementation for self-supervised speech representation learning.

    This model processes raw audio waveforms through a convolutional feature extractor,
    applies masking, and feeds the representations into a Transformer encoder to learn
    contextualized speech representations.

    Args:
        cfg (object): Configuration object containing model hyperparameters.
        num_classes (int): Total number of target classes/labels for token generation.
    """

    def __init__(
        self, 
        cfg, 
        num_classes
    ):
        super().__init__()
        feature_enc_layers = eval(cfg.conv_feature_layers) # Parse the convolutional layer configurations from a string representation of a list
        # Determine the embedding dimension output by the final convolutional layer
        self.embed = feature_enc_layers[-1][0]
        # Convolutional feature extractor to process raw audio input
        self.feature_extractor = ConvFeatureExtractionModel(conv_layers=feature_enc_layers, dropout=0.0, mode=cfg.extractor_mode, conv_bias=cfg.conv_bias)
        # Projection layer if the convolutional embedding dimension differs from the transformer dimension
        self.post_extract_proj = nn.Linear(self.embed, cfg.encoder_embed_dim) if self.embed != cfg.encoder_embed_dim else None
        # Dropout layers for regularization
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)
        # Gradient multiplier for the convolutional feature extractor (used to scale or block gradients)
        self.feature_grad_mult = cfg.feature_grad_mult
        # Determine the final projection dimension
        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        # Learnable mask embedding token used to replace masked frames during pre-training
        self.mask_emb = nn.Parameter(torch.FloatTensor(cfg.encoder_embed_dim).uniform_())
        # The core Transformer Encoder network
        self.encoder = TransformerEncoder(cfg)
        # Layer normalization applied to the extracted convolutional features
        self.layer_norm = nn.LayerNorm(self.embed, eps=1e-5, elementwise_affine=True)
        # Optional Gated Linear Unit (GLU) for target projections
        self.target_glu = None
        if cfg.target_glu: self.target_glu = nn.Sequential(nn.Linear(final_dim, final_dim * 2), nn.GLU())
        self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim) # Final linear projection layer
        # Learnable label embeddings concatenated for cluster/target classification
        self.label_embs_concat = nn.Parameter(torch.FloatTensor(sum([num_classes]), final_dim))
        nn.init.uniform_(self.label_embs_concat)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Upgrades a given state dict named for backward compatibility.

        Args:
            state_dict (dict): The state dictionary to upgrade.
            name (str): The name of the model component.

        Returns:
            dict: The updated state dictionary.
        """

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def forward_features(self, source):
        """
        Extracts features from the raw audio source using the convolutional feature extractor.

        Args:
            source (Tensor): Input raw audio waveform tensor of shape `(batch, samples)`.

        Returns:
            Tensor: Extracted features tensor.
        """

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            # Scale gradients if feature_grad_mult is explicitly set and not equal to 1.0
            if self.feature_grad_mult != 1.0: features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            # Disable gradient computation for the feature extractor if multiplier is 0 or negative
            with torch.no_grad():
                features = self.feature_extractor(source)

        return features

    def forward_padding_mask(self, features, padding_mask):
        """
        Aligns and downsamples the attention padding mask to match the feature extractor's output length.

        Args:
            features (Tensor): Features tensor of shape `(batch, frames, channels)`.
            padding_mask (Tensor): Original boolean padding mask for the raw audio.

        Returns:
            Tensor: Downsampled boolean padding mask of shape `(batch, frames)`.
        """

        # Calculate the remainder when matching the raw mask length with the downsampled feature length
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0: padding_mask = padding_mask[:, :-extra]

        return padding_mask.view(padding_mask.size(0), features.size(1), -1).all(-1)

    def forward(self, source, padding_mask = None, output_layer = None):
        """
        Forward pass of the HuBERT model.

        Args:
            source (Tensor): Input raw audio waveform tensor of shape `(batch, samples)`.
            padding_mask (Tensor, optional): Boolean padding mask for input. Defaults to None.
            output_layer (int, optional): Specific Transformer layer index to output from. Defaults to None.

        Returns:
            - "x" (Tensor): The encoder's contextualized output representations.
            - "padding_mask" (Tensor): The downsampled padding mask applied.
            - "features" (Tensor): The projected convolutional features before the encoder.
        """
    
        # 1. Feature Extraction & Normalization
        features = self.forward_features(source)
        # Transpose to (batch, frames, channels) before applying LayerNorm
        features = self.layer_norm(features.transpose(1, 2))

        # Keep a copy of unmasked features (often used for computing target losses)
        unmasked_features = features.clone()
        # 2. Process Padding Mask
        padding_mask = self.forward_padding_mask(features, padding_mask)

        # 3. Projection & Regularization
        if self.post_extract_proj is not None: features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        # 4. Transformer Encoding
        x, _ = self.encoder(features, padding_mask=padding_mask, layer=None if output_layer is None else output_layer - 1)
        return {"x": x, "padding_mask": padding_mask, "features": features}

    def extract_features(self, source, ret_conv = False, output_layer = None):
        """
        Utility method to extract either convolutional or contextualized Transformer features.

        Args:
            source (Tensor): Input raw audio waveform tensor.
            ret_conv (bool, optional): If True, returns CNN features; otherwise returns Transformer features. Defaults to False.
            output_layer (int, optional): Specific Transformer layer index to extract from. Defaults to None.

        Returns:
            tuple: (features_tensor, padding_mask)
        """

        # Create a default empty padding mask (all False) matching the source shape
        padding_mask = torch.BoolTensor(source.shape).fill_(False).to(source.device)
        # Execute standard forward pass
        res = self.forward(source, padding_mask=padding_mask, output_layer=output_layer)
        # Return either the CNN projected features or the contextualized Transformer representations
        return res["features"] if ret_conv else res["x"], res["padding_mask"]