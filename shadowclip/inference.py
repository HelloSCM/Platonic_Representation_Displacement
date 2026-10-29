"""Encode unseen images and text without an LLM teacher."""

import torch

from representation.extraction.vision import VisionEncoder

from .checkpoint import load_core


class Encoder:
    """Load frozen CLIP/DINO backbones and exported ShadowCLIP adapters."""

    def __init__(
        self, checkpoint, device="cuda", dtype="float32", clip_pretrained=None, dino_checkpoint=None
    ):
        """Load trained adapters and the frozen CLIP and optional DINO backbones."""
        import open_clip

        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.core = load_core(checkpoint, self.device)
        info = self.core.config["encoders"]
        clip = info["clip_text"]
        self.clip, _, self.transform = open_clip.create_model_and_transforms(
            clip["model"], pretrained=clip_pretrained or clip["pretrained"]
        )
        self.clip.visual.proj = None
        self.clip.text_projection = None
        self.clip.eval().to(self.device).requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(clip["model"])
        self.dino = None
        if self.core.vision is not None:
            dino = info["dino"]
            self.dino = VisionEncoder(
                dino["model"],
                device=str(self.device),
                layers="final",
                pool="auto" if dino.get("pool") == "model" else dino.get("pool", "auto"),
                dtype=dtype,
                checkpoint=dino_checkpoint or dino.get("checkpoint"),
            )

    @torch.inference_mode()
    def encode_image(self, images):
        """Encode images into normalized embeddings using CLIP and optional DINO features."""
        clip_inputs = torch.stack([self.transform(image.convert("RGB")) for image in images])
        dino = None
        if self.dino is not None:
            dino_inputs = torch.stack(
                [self.dino.transform(image.convert("RGB")) for image in images]
            )
            dino = self.dino(dino_inputs)["final"]
        with torch.autocast(
            self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32
        ):
            clip = self.clip.encode_image(clip_inputs.to(self.device), normalize=False)
            return self.core.encode_image(clip, dino)

    @torch.inference_mode()
    def encode_text(self, texts):
        """Encode text into normalized embeddings using the adapted CLIP text branch."""
        tokens = self.tokenizer(texts).to(self.device)
        with torch.autocast(
            self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32
        ):
            clip = self.clip.encode_text(tokens, normalize=False)
            return self.core.encode_text(clip)
