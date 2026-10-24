"""Adapter around the authors' unmodified DDColor implementation."""
import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
OFFICIAL_REPO = ROOT / "third_party" / "DDColor"
if str(OFFICIAL_REPO) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO))

try:
    from basicsr.archs.ddcolor_arch import DDColor as OfficialDDColor
except ImportError as exc:
    raise ImportError(f"Cannot import official DDColor from {OFFICIAL_REPO}") from exc


def neutral_lab_l_to_srgb(luminance):
    """Convert normalized CIE L* to the neutral sRGB input expected by DDColor."""
    l_star = (luminance + 1.0) * 50.0
    f = (l_star + 16.0) / 116.0
    linear = torch.where(l_star > 8.0, f.pow(3), l_star / 903.3)
    srgb = torch.where(linear <= 0.0031308, 12.92 * linear,
                       1.055 * torch.clamp(linear, min=0).pow(1.0 / 2.4) - 0.055)
    return srgb.clamp(0, 1).repeat(1, 3, 1, 1)


class DDColorAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        size = config["image_size"]
        self.model = OfficialDDColor(
            encoder_name=config.get("encoder_name", "convnext-t"),
            decoder_name="MultiScaleColorDecoder",
            input_size=(size, size),
            nf=config.get("nf", 512),
            num_output_channels=2,
            last_norm="Spectral",
            do_normalize=False,
            num_queries=config.get("num_queries", 100),
            num_scales=3,
            dec_layers=config.get("decoder_layers", 9),
            encoder_from_pretrain=False,
        )

    def forward(self, luminance):
        # Official DDColor predicts CIE a*b* in approximately [-128, 128].
        return self.model(neutral_lab_l_to_srgb(luminance)) / 128.0

    def load_official(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
        state = {key.removeprefix("module."): value for key, value in state.items()}
        incompatible = self.model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(f"Official checkpoint mismatch: {incompatible}")

    @property
    def encoder(self):
        return self.model.encoder

