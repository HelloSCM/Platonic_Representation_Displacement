"""Extract transformer language features and LLM2Vec-Gen embeddings."""

import torch

from .common import BlockHooks, pool_tokens, set_seed


def bidirectional_mask(module, args, kwargs):
    """Replace causal masking with a padding-only additive attention mask."""
    mask = kwargs["attention_mask"]
    dtype = module.get_input_embeddings().weight.dtype
    padding = torch.zeros_like(mask, dtype=dtype).masked_fill(~mask.bool(), torch.finfo(dtype).min)
    kwargs["attention_mask"] = padding[:, None, None, :].expand(-1, 1, mask.shape[1], -1)
    kwargs["use_cache"] = False
    return args, kwargs


def enable_bidirectional(model):
    """Enable bidirectional SDPA for the frozen LLM encoder."""
    model.set_attn_implementation("sdpa")
    for module in model.modules():
        if hasattr(module, "config"):
            module.config.is_causal = False
        if hasattr(module, "is_causal"):
            module.is_causal = False
    return model.register_forward_pre_hook(bidirectional_mask, with_kwargs=True)


class LanguageEncoder:
    """Mean-pool valid tokens after complete language transformer blocks."""

    def __init__(
        self,
        model,
        device="cuda",
        layers="all",
        dtype="float32",
        max_length=128,
        trust_remote_code=False,
        random=False,
        seed=42,
    ):
        """Initialize a frozen language model, tokenizer, and optional block hooks."""
        from transformers import AutoConfig, AutoModel, AutoTokenizer, CLIPTextModel

        set_seed(seed)
        self.device = torch.device(device)
        self.max_length = max_length
        self.layers = layers
        config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        is_clip = config.model_type in {"clip", "clip_text_model"}
        model_cls = CLIPTextModel if is_clip else AutoModel
        if random:
            text_config = config.text_config if config.model_type == "clip" else config
            self.model = (
                model_cls(text_config)
                if is_clip
                else AutoModel.from_config(config, trust_remote_code=trust_remote_code)
            )
        else:
            self.model = model_cls.from_pretrained(
                model, dtype=getattr(torch, dtype), trust_remote_code=trust_remote_code
            )
        self.model.eval().to(self.device, dtype=getattr(torch, dtype)).requires_grad_(False)
        if is_clip:
            text_model = getattr(self.model, "text_model", self.model)
            blocks = text_model.encoder.layers
            self.max_length = min(max_length, self.model.config.max_position_embeddings)
        elif config.model_type == "bloom":
            blocks = self.model.h
        else:
            blocks = self.model.layers
        self.mask = None
        self.hooks = (
            None
            if layers == "final"
            else BlockHooks(blocks, layers, lambda x: pool_tokens(x, "mean", mask=self.mask))
        )
        self.metadata = {
            "backend": "transformers",
            "model": model,
            "model_type": config.model_type,
            "layers": layers,
            "pool": "mean",
            "max_length": self.max_length,
            "dtype": dtype,
            "random": random,
            "seed": seed,
            "normalized": False,
        }
        self.assets = {}

    @torch.inference_mode()
    def __call__(self, texts):
        """Tokenize text and return pooled features from the selected layers."""
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
        ).to(self.device)
        self.mask = tokens["attention_mask"]
        if self.hooks:
            self.hooks.clear()
        kwargs = (
            {}
            if self.metadata["model_type"] in {"clip", "clip_text_model"}
            else {"use_cache": False}
        )
        output = self.model(**tokens, **kwargs)
        if self.hooks:
            return dict(self.hooks.features)
        return {"final": pool_tokens(output.last_hidden_state, "mean", mask=self.mask)}

    def close(self):
        """Remove the language model's feature capture hooks."""
        if self.hooks:
            self.hooks.close()


class LLM2VecEncoder:
    """Encode final embeddings using LLM2Vec-Gen."""

    def __init__(self, model, device="cuda", dtype="float32", max_length=512, instruction=""):
        """Load a frozen LLM2Vec-Gen encoder with bidirectional attention."""
        from llm2vec_gen import LLM2VecGenModel
        from llm2vec_gen.models import EncoderDecoderModel
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model)
        if not hasattr(tokenizer, "additional_special_tokens"):
            tokenizer.additional_special_tokens = tokenizer.extra_special_tokens
        backbone = EncoderDecoderModel.from_pretrained(model, causal_lm_decoder=False)
        self.attention_hook = enable_bidirectional(backbone.encoder_model.get_base_model())
        backbone.eval().to(dtype=getattr(torch, dtype)).requires_grad_(False)
        self.model = LLM2VecGenModel(backbone, tokenizer, device=device)
        self.max_length = max_length
        self.instruction = instruction
        self.metadata = {
            "backend": "llm2vec",
            "bidirectional": True,
            "model": model,
            "layers": "final",
            "max_length": max_length,
            "instruction": instruction,
            "dtype": dtype,
            "normalized": False,
        }
        self.assets = {}

    @torch.inference_mode()
    def __call__(self, texts):
        """Encode text with the configured instruction prefix and token limit."""
        x = self.model.encode(
            [self.instruction + text for text in texts], max_length=self.max_length
        )
        return {"final": x}

    def close(self):
        """Remove the attention mask hook from the encoder."""
        self.attention_hook.remove()
