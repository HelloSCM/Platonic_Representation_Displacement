"""Cache final CLIP backbone features and their projection matrices."""

import torch
from torch import nn


class CLIPEncoder:
    """Extract unnormalized hidden or projected OpenCLIP features."""

    def __init__(
        self, model, pretrained, modality, device="cuda", dtype="float32", projected=False
    ):
        """Load a frozen OpenCLIP encoder and retain its projection assets."""
        import open_clip

        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.modality = modality
        self.projected = projected
        self.model, _, self.transform = open_clip.create_model_and_transforms(
            model, pretrained=pretrained
        )
        self.model.eval().to(self.device).requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(model)
        proj = self.model.visual.proj if modality == "vision" else self.model.text_projection
        matrix = proj.weight.T if isinstance(proj, nn.Linear) else proj
        self.assets = {
            modality: matrix.detach().float().cpu().numpy(),
            "logit_scale": self.model.logit_scale.detach().float().cpu().numpy(),
        }
        if not projected:
            if modality == "vision":
                self.model.visual.proj = None
            else:
                self.model.text_projection = None
        self.metadata = {
            "backend": "openclip",
            "model": model,
            "pretrained": pretrained,
            "layers": "final",
            "pool": "model",
            "dtype": dtype,
            "projected": projected,
            "normalized": False,
        }

    @torch.inference_mode()
    def __call__(self, inputs):
        """Encode an image or text batch into final CLIP features."""
        with torch.autocast(
            self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32
        ):
            if self.modality == "vision":
                x = self.model.encode_image(inputs.to(self.device), normalize=False)
            else:
                tokens = self.tokenizer(inputs).to(self.device)
                x = self.model.encode_text(tokens, normalize=False)
        return {"final": x}

    def close(self):
        """Provide the cleanup interface used by the extraction pipeline."""
        pass
