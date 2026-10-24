import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid()
        )
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, x):
        x = x * self.channel(x)
        spatial = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), dim=1)
        return x * self.spatial(spatial)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(16, out_channels), out_channels), nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(16, out_channels), out_channels), nn.SiLU(inplace=True), CBAM(out_channels)
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.layers(torch.cat((x, skip), dim=1))


class SemanticColorizer(nn.Module):
    def __init__(self, backbone="resnet34", pretrained_encoder=True):
        super().__init__()
        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained_encoder else None)
        elif backbone == "resnet34":
            net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained_encoder else None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.backbone_name = backbone
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool, self.layer1, self.layer2, self.layer3, self.layer4 = (
            net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4
        )
        self.d4 = DecoderBlock(512, 256, 256)
        self.d3 = DecoderBlock(256, 128, 128)
        self.d2 = DecoderBlock(128, 64, 64)
        self.d1 = DecoderBlock(64, 64, 32)
        self.d0 = DecoderBlock(32, 1, 32)
        self.head = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(inplace=True),
                                  nn.Conv2d(32, 2, 1), nn.Tanh())
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def encoder_parameters(self):
        """Yield encoder parameters without registering a duplicate module tree."""
        for module in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            yield from module.parameters()

    def forward(self, luminance):
        gray = ((luminance + 1.0) * 0.5).repeat(1, 3, 1, 1)
        x0 = self.stem((gray - self.mean) / self.std)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return self.head(self.d0(self.d1(self.d2(self.d3(self.d4(x4, x3), x2), x1), x0), luminance))


class EdgeExtractor(nn.Module):
    """Trích xuất biên cạnh độ sáng bằng bộ lọc đạo hàm không gian Scharr khả vi."""

    def __init__(self, out_channels=16):
        super().__init__()
        kx = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]], dtype=torch.float32) / 16.0
        ky = torch.tensor([[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]], dtype=torch.float32) / 16.0
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(4, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(4, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, luminance):
        gx = F.conv2d(luminance, self.kx, padding=1)
        gy = F.conv2d(luminance, self.ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return self.edge_conv(mag)


class StripPooling(nn.Module):
    """Mở rộng trường tiếp nhận toàn cảnh (bầu trời, phong cảnh) theo 2 trục ngang và dọc."""

    def __init__(self, channels):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.pool_w = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_h = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(min(16, channels), channels),
            nn.SiLU(inplace=True),
        )
        self.conv_w = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(min(16, channels), channels),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        x_h = F.interpolate(self.conv_h(self.pool_h(x)), size=(h, w), mode="bilinear", align_corners=False)
        x_w = F.interpolate(self.conv_w(self.pool_w(x)), size=(h, w), mode="bilinear", align_corners=False)
        return x * (1.0 + self.gate(x_h + x_w))


class EdgeGatedDecoderBlock(nn.Module):
    """Decoder block kết hợp Skip Connection và cổng hướng dẫn biên cạnh (Edge Gate)."""

    def __init__(self, in_channels, skip_channels, out_channels, edge_channels=0):
        super().__init__()
        total_in = in_channels + skip_channels + edge_channels
        self.layers = nn.Sequential(
            nn.Conv2d(total_in, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.SiLU(inplace=True),
            CBAM(out_channels),
        )
        if edge_channels > 0:
            self.edge_gate = nn.Sequential(
                nn.Conv2d(edge_channels, out_channels, 1, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.edge_gate = None

    def forward(self, x, skip, edge=None):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if edge is not None:
            if edge.shape[-2:] != skip.shape[-2:]:
                edge = F.interpolate(edge, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            fused = self.layers(torch.cat((x, skip, edge), dim=1))
            if self.edge_gate is not None:
                fused = fused * (1.0 + self.edge_gate(edge))
            return fused
        return self.layers(torch.cat((x, skip), dim=1))



class ColorPaletteCrossAttention(nn.Module):
    """Mô-đun Chú ý Chéo Bảng Màu (Color Palette Cross-Attention - CPCA).
    Chứa K=32 vector truy vấn màu chuẩn tắc (Sky Blue, Tropical Foliage, Olive Khaki, Warm Skin, Red Banner, v.v.).
    Cho phép các đặc trưng không gian truy vấn và gắn kết trực tiếp với các cụm màu chuẩn,
    ngăn ngừa hiện tượng suy thoái về màu xám nâu/sepia mờ nhạt (regression-to-the-mean)."""

    def __init__(self, in_channels=512, num_queries=32):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, in_channels) * 0.02)
        self.q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.k_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.v_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.out_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Dropout(0.05),
        )
        self.norm = nn.GroupNorm(min(16, in_channels), in_channels)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)
        q_pal = self.queries.unsqueeze(0).expand(b, -1, -1)
        q = self.q_proj(x_flat)
        k = self.k_proj(q_pal)
        v = self.v_proj(q_pal)
        scale = c ** -0.5
        attn = torch.bmm(q, k.transpose(1, 2)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)
        out = self.out_proj(out).permute(0, 2, 1).view(b, c, h, w)
        return x + self.gamma * self.norm(out)


class VietColorNet(nn.Module):
    """Mạng tô màu ảnh lịch sử Việt Nam hướng biên cạnh và phân tách Sắc thái - Độ đậm màu."""

    def __init__(self, backbone="resnet34", pretrained_encoder=True, chroma_boost=1.15,
                 use_palette_attention=False, num_palette_queries=32,
                 decoupled_hue_chroma=False, max_chroma=48.0,
                 chroma_calibration=False, chroma_residual_scale=0.7,
                 use_global_chroma_scale=False):
        super().__init__()
        self.backbone_name = backbone
        self.chroma_boost = float(chroma_boost)
        self.use_palette_attention = bool(use_palette_attention)
        self.decoupled_hue_chroma = bool(decoupled_hue_chroma)
        self.max_chroma = float(max_chroma) / 128.0
        self.chroma_calibration = bool(chroma_calibration)
        self.chroma_residual_scale = float(chroma_residual_scale)
        self.use_global_chroma_scale = bool(use_global_chroma_scale)
        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained_encoder else None)
        elif backbone == "resnet34":
            net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained_encoder else None)
        else:
            raise ValueError(f"Unsupported VietColorNet backbone: {backbone}")

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool, self.layer1, self.layer2, self.layer3, self.layer4 = (
            net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4
        )

        # 1. Edge-Guidance Stream
        self.edge_stream = EdgeExtractor(out_channels=16)

        # 2. Context Bridge
        self.context = StripPooling(512)
        if self.use_palette_attention:
            self.palette_attention = ColorPaletteCrossAttention(512, num_queries=num_palette_queries)
        else:
            self.palette_attention = None

        # 3. Edge-Gated Decoders
        self.d4 = EdgeGatedDecoderBlock(512, 256, 256)
        self.d3 = EdgeGatedDecoderBlock(256, 128, 128)
        self.d2 = EdgeGatedDecoderBlock(128, 64, 64)
        self.d1 = EdgeGatedDecoderBlock(64, 64, 32)
        self.d0 = EdgeGatedDecoderBlock(32, 1, 32, edge_channels=16)

        # 4. Decoupled Hue-Chroma Heads
        self.hue_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 2, 1),
        )
        self.chroma_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        if self.decoupled_hue_chroma:
            # Tên mới ngăn gate V4 bị hiểu nhầm là magnitude head của V5 khi
            # nạp shared trunk bằng strict=False.
            self.chroma_magnitude_head = nn.Sequential(
                nn.Conv2d(32, 16, 3, padding=1, bias=False),
                nn.GroupNorm(4, 16),
                nn.SiLU(inplace=True),
                nn.Conv2d(16, 1, 1),
                nn.Sigmoid(),
            )
            # sigmoid(-1.1) ~= 0.25: bắt đầu quanh chroma 12 khi max_chroma=48,
            # thay vì mặc định 24 gây quá bão hòa ngay trước fine-tune.
            nn.init.constant_(self.chroma_magnitude_head[-2].bias, -1.1)
            if self.chroma_calibration:
                self.chroma_residual_head = nn.Conv2d(32, 1, 1)
                self.neutral_confidence_head = nn.Conv2d(32, 1, 1)
                nn.init.zeros_(self.chroma_residual_head.weight)
                nn.init.zeros_(self.chroma_residual_head.bias)
                nn.init.zeros_(self.neutral_confidence_head.weight)
                nn.init.constant_(self.neutral_confidence_head.bias, 6.0)
                if self.use_global_chroma_scale:
                    self.global_chroma_head = nn.Linear(512, 1)
                    nn.init.zeros_(self.global_chroma_head.weight)
                    nn.init.zeros_(self.global_chroma_head.bias)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, luminance):
        # 1. Trích xuất đặc trưng biên
        edge_feat = self.edge_stream(luminance)

        # 2. Trích xuất đặc trưng ngữ nghĩa đa tầng
        gray = ((luminance + 1.0) * 0.5).repeat(1, 3, 1, 1)
        x0 = self.stem((gray - self.mean) / self.std)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.context(self.layer4(x3))
        if self.palette_attention is not None:
            x4 = self.palette_attention(x4)
        global_chroma_scale = None
        if self.decoupled_hue_chroma and self.chroma_calibration and self.use_global_chroma_scale:
            pooled = F.adaptive_avg_pool2d(x4, 1).flatten(1)
            global_chroma_scale = 1.0 + self.chroma_residual_scale * torch.tanh(
                self.global_chroma_head(pooled)
            ).view(-1, 1, 1, 1)

        # 3. Giải mã có kiểm soát biên
        y = self.d4(x4, x3)
        y = self.d3(y, x2)
        y = self.d2(y, x1)
        y = self.d1(y, x0)
        y = self.d0(y, luminance, edge=edge_feat)

        # 4. Phân tách Sắc độ (Hue/ab) và Độ đậm màu (Chroma Saliency)
        # Sử dụng tanh để ánh xạ liên tục không gian Lab [-1, 1], loại bỏ điểm kỳ dị ở vùng xám trung tính
        hue_vector = torch.tanh(self.hue_head(y))
        if self.decoupled_hue_chroma:
            hue_direction = F.normalize(hue_vector, p=2, dim=1, eps=1e-6)
            chroma = self.chroma_magnitude_head(y) * self.max_chroma
            if self.chroma_calibration:
                residual = 1.0 + self.chroma_residual_scale * torch.tanh(self.chroma_residual_head(y))
                confidence = torch.sigmoid(self.neutral_confidence_head(y))
                chroma = torch.clamp(chroma * residual * confidence, 0.0, self.max_chroma)
                if global_chroma_scale is not None:
                    chroma = torch.clamp(chroma * global_chroma_scale, 0.0, self.max_chroma)
            return torch.clamp(hue_direction * chroma, -1.0, 1.0)

        base_ab = hue_vector
        chroma_gate = self.chroma_head(y)
        # Tăng cường độ bão hòa có chọn lọc cho vùng màu (lá cây, da người, cờ, quân phục),
        # bảo toàn độ sạch của các vùng xám/trung tính không bị ám sắc nâu/sepia.
        boost_factor = max(self.chroma_boost - 1.0, 0.0)
        ab_pred = torch.clamp(base_ab * (1.0 + boost_factor * chroma_gate), -1.0, 1.0)
        return ab_pred


class DINOv2SemanticBridge(nn.Module):
    """Frozen DINOv2-S/14 features, gated into the V6 spatial bottleneck.

    The gate starts at zero so loading a V6 checkpoint preserves its behaviour
    before V7 fine-tuning.  The official Meta torch-hub implementation is used
    and its weights are saved inside the resulting V7 checkpoint.
    """

    def __init__(self, out_channels=512, pretrained=True, frozen=True,
                 gate_mode="zero_scalar", gate_init_bias=-3.0,
                 trainable_blocks=0):
        super().__init__()
        hub_root = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        if hub_root.is_dir():
            self.encoder = torch.hub.load(str(hub_root), "dinov2_vits14",
                                          source="local", pretrained=pretrained)
        else:
            self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                          pretrained=pretrained)
        self.trainable_blocks = int(trainable_blocks)
        self.frozen = bool(frozen) and self.trainable_blocks <= 0
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        if self.trainable_blocks > 0:
            for block in self.encoder.blocks[-self.trainable_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            for parameter in self.encoder.norm.parameters():
                parameter.requires_grad_(True)
        self.project = nn.Sequential(
            nn.Conv2d(384, out_channels, 1, bias=False),
            nn.GroupNorm(16, out_channels),
            nn.SiLU(inplace=True),
        )
        self.gate_mode = gate_mode
        if gate_mode == "channel_sigmoid":
            self.gate_logits = nn.Parameter(torch.full((1, out_channels, 1, 1), float(gate_init_bias)))
            self.register_buffer("gate_warmup_scale", torch.tensor(1.0))
        elif gate_mode == "zero_scalar":
            self.gamma = nn.Parameter(torch.zeros(()))
        else:
            raise ValueError(f"Unsupported DINO gate mode: {gate_mode}")
        self.register_buffer("dino_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("dino_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, gray, output_size):
        # 252 is divisible by patch size 14 and is close to the 256px training crop.
        image = F.interpolate(gray, size=(252, 252), mode="bilinear", align_corners=False)
        image = (image - self.dino_mean) / self.dino_std
        if self.frozen:
            with torch.no_grad():
                tokens = self.encoder.get_intermediate_layers(image, n=1, reshape=True)[0]
        else:
            tokens = self.encoder.get_intermediate_layers(image, n=1, reshape=True)[0]
        feature = self.project(tokens)
        feature = F.interpolate(feature, size=output_size, mode="bilinear", align_corners=False)
        if self.gate_mode == "channel_sigmoid":
            return self.gate_warmup_scale * torch.sigmoid(self.gate_logits) * feature
        return torch.tanh(self.gamma) * feature

    def set_warmup_scale(self, value):
        if self.gate_mode == "channel_sigmoid":
            self.gate_warmup_scale.fill_(float(max(0.0, min(1.0, value))))


class VietColorNetV7(VietColorNet):
    """V7: V6 decoder + DINOv2 semantic bridge + residual Palette MoE.

    Residual experts and semantic fusion are zero-initialised, which makes a
    partially loaded V6 checkpoint a stable and auditable V7 starting point.
    """

    def __init__(self, backbone="resnet34", pretrained_encoder=False,
                 num_palette_queries=32, max_chroma=48.0,
                 chroma_residual_scale=0.7, use_global_chroma_scale=True,
                 dino_pretrained=True, dino_frozen=True, num_palette_experts=3,
                 use_palette_moe=True, semantic_gate_mode="zero_scalar",
                 semantic_gate_init_bias=-3.0, semantic_gate_warmup_epochs=0,
                 dino_trainable_blocks=0):
        super().__init__(
            backbone=backbone, pretrained_encoder=pretrained_encoder,
            chroma_boost=1.0, use_palette_attention=True,
            num_palette_queries=num_palette_queries, decoupled_hue_chroma=True,
            max_chroma=max_chroma, chroma_calibration=True,
            chroma_residual_scale=chroma_residual_scale,
            use_global_chroma_scale=use_global_chroma_scale,
        )
        self.semantic_bridge = DINOv2SemanticBridge(
            512, pretrained=dino_pretrained, frozen=dino_frozen,
            gate_mode=semantic_gate_mode, gate_init_bias=semantic_gate_init_bias,
            trainable_blocks=dino_trainable_blocks)
        self.semantic_gate_warmup_epochs = int(semantic_gate_warmup_epochs)
        self.use_palette_moe = bool(use_palette_moe)
        self.num_palette_experts = int(num_palette_experts)
        if self.use_palette_moe:
            self.palette_gate = nn.Conv2d(32, self.num_palette_experts, 1)
            self.palette_experts = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(32, 24, 3, padding=1, bias=False),
                    nn.GroupNorm(6, 24), nn.SiLU(inplace=True),
                    nn.Conv2d(24, 2, 1), nn.Tanh(),
                ) for _ in range(self.num_palette_experts)
            ])
            nn.init.zeros_(self.palette_gate.weight)
            nn.init.zeros_(self.palette_gate.bias)
            for expert in self.palette_experts:
                nn.init.zeros_(expert[-2].weight)
                nn.init.zeros_(expert[-2].bias)
            self.moe_residual_scale = nn.Parameter(torch.tensor(0.05))

    def encoder_parameters(self):
        for module in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            yield from module.parameters()

    def semantic_parameters(self):
        yield from self.semantic_bridge.project.parameters()
        if hasattr(self.semantic_bridge, "gamma"):
            yield self.semantic_bridge.gamma
        if hasattr(self.semantic_bridge, "gate_logits"):
            yield self.semantic_bridge.gate_logits

    def set_training_epoch(self, epoch):
        if self.semantic_gate_warmup_epochs > 0:
            self.semantic_bridge.set_warmup_scale((epoch + 1) / self.semantic_gate_warmup_epochs)
        else:
            self.semantic_bridge.set_warmup_scale(1.0)

    def decode_features(self, luminance):
        """Shared encoder/decoder trunk: 32-channel full-resolution features and global scale."""
        edge_feat = self.edge_stream(luminance)
        gray = ((luminance + 1.0) * 0.5).repeat(1, 3, 1, 1)
        x0 = self.stem((gray - self.mean) / self.std)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.context(self.layer4(x3))
        x4 = x4 + self.semantic_bridge(gray, x4.shape[-2:])
        if self.palette_attention is not None:
            x4 = self.palette_attention(x4)
        global_scale = None
        if self.use_global_chroma_scale:
            pooled = F.adaptive_avg_pool2d(x4, 1).flatten(1)
            global_scale = 1.0 + self.chroma_residual_scale * torch.tanh(
                self.global_chroma_head(pooled)).view(-1, 1, 1, 1)
        y = self.d4(x4, x3)
        y = self.d3(y, x2)
        y = self.d2(y, x1)
        y = self.d1(y, x0)
        y = self.d0(y, luminance, edge=edge_feat)
        return y, global_scale

    def forward(self, luminance):
        y, global_scale = self.decode_features(luminance)
        hue = F.normalize(torch.tanh(self.hue_head(y)), p=2, dim=1, eps=1e-6)
        chroma = self.chroma_magnitude_head(y) * self.max_chroma
        residual = 1.0 + self.chroma_residual_scale * torch.tanh(self.chroma_residual_head(y))
        confidence = torch.sigmoid(self.neutral_confidence_head(y))
        chroma = torch.clamp(chroma * residual * confidence, 0.0, self.max_chroma)
        if global_scale is not None:
            chroma = torch.clamp(chroma * global_scale, 0.0, self.max_chroma)
        prediction = hue * chroma
        if self.use_palette_moe:
            gates = F.softmax(self.palette_gate(y), dim=1)
            expert_outputs = torch.stack([expert(y) for expert in self.palette_experts], dim=1)
            mixture = (expert_outputs * gates.unsqueeze(2)).sum(dim=1)
            prediction = prediction + torch.tanh(self.moe_residual_scale) * mixture
        return torch.clamp(prediction, -1.0, 1.0)


class VietColorNetV14(VietColorNetV7):
    """V14: the V7 trunk with a classification colour head.

    Regression heads (V5-V13) minimise distance to the true ab, so when an object
    may be red or green they predict the mean of the two, a desaturated yellow-brown.
    V14 predicts a probability for each of ``num_color_bins`` quantised ab colours
    (Zhang et al., "Colorful Image Colorization", ECCV 2016) and decodes it with the
    annealed mean: sharpen the distribution with temperature T, then take its mean,
    which keeps the chosen mode vivid without the speckle of a plain argmax.
    Logits are computed at half resolution to fit a 6 GB GPU at 512 px.
    """

    def __init__(self, num_color_bins=259, temperature=0.38, logits_stride=2, **kwargs):
        super().__init__(**kwargs)
        self.num_color_bins = int(num_color_bins)
        self.temperature = float(temperature)
        self.logits_stride = int(logits_stride)
        self.color_logits_head = nn.Sequential(
            nn.Conv2d(32, 96, 3, padding=1, bias=False), nn.GroupNorm(8, 96), nn.SiLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1, bias=False), nn.GroupNorm(8, 96), nn.SiLU(inplace=True),
            nn.Conv2d(96, self.num_color_bins, 1),
        )
        # Normalised ab centre of each colour class; filled by set_color_bins() before training
        # and restored from the checkpoint at inference.
        self.register_buffer("color_bin_centers", torch.zeros(self.num_color_bins, 2))
        self.last_color_logits = None

    def set_color_bins(self, centers):
        centers = torch.as_tensor(centers, dtype=torch.float32)
        if centers.shape != self.color_bin_centers.shape:
            raise ValueError(f"Expected {tuple(self.color_bin_centers.shape)} colour bins, got {tuple(centers.shape)}")
        self.color_bin_centers.copy_(centers.to(self.color_bin_centers.device))

    def decode_logits(self, logits, temperature=None):
        temperature = self.temperature if temperature is None else temperature
        probability = F.softmax(logits.float() / temperature, dim=1)
        return torch.einsum("bqhw,qc->bchw", probability, self.color_bin_centers)

    def forward(self, luminance):
        y, _ = self.decode_features(luminance)
        if self.logits_stride > 1:
            y = F.avg_pool2d(y, self.logits_stride)
        logits = self.color_logits_head(y)
        self.last_color_logits = logits
        ab = self.decode_logits(logits)
        ab = F.interpolate(ab, size=luminance.shape[-2:], mode="bilinear", align_corners=False)
        return torch.clamp(ab, -1.0, 1.0)


def build_model(config):
    if config["model"] == "semantic_resnet":
        return SemanticColorizer(config["backbone"], config.get("pretrained_encoder", True))
    if config["model"] == "vietcolornet":
        return VietColorNet(
            backbone=config.get("backbone", "resnet34"),
            pretrained_encoder=config.get("pretrained_encoder", True),
            chroma_boost=config.get("chroma_boost", 1.15),
            use_palette_attention=config.get("use_palette_attention", False),
            num_palette_queries=config.get("num_palette_queries", 32),
        )
    if config["model"] == "vietcolornet_v5":
        return VietColorNet(
            backbone=config.get("backbone", "resnet34"),
            pretrained_encoder=config.get("pretrained_encoder", True),
            chroma_boost=1.0,
            use_palette_attention=config.get("use_palette_attention", True),
            num_palette_queries=config.get("num_palette_queries", 32),
            decoupled_hue_chroma=True,
            max_chroma=config.get("max_chroma", 48.0),
        )
    if config["model"] == "vietcolornet_v6":
        return VietColorNet(
            backbone=config.get("backbone", "resnet34"),
            pretrained_encoder=config.get("pretrained_encoder", True),
            chroma_boost=1.0,
            use_palette_attention=config.get("use_palette_attention", True),
            num_palette_queries=config.get("num_palette_queries", 32),
            decoupled_hue_chroma=True,
            max_chroma=config.get("max_chroma", 48.0),
            chroma_calibration=True,
            chroma_residual_scale=config.get("chroma_residual_scale", 0.7),
            use_global_chroma_scale=config.get("use_global_chroma_scale", False),
        )
    if config["model"] in {"vietcolornet_v7", "vietcolornet_v8"}:
        return VietColorNetV7(
            backbone=config.get("backbone", "resnet34"),
            pretrained_encoder=config.get("pretrained_encoder", False),
            num_palette_queries=config.get("num_palette_queries", 32),
            max_chroma=config.get("max_chroma", 48.0),
            chroma_residual_scale=config.get("chroma_residual_scale", 0.7),
            use_global_chroma_scale=config.get("use_global_chroma_scale", True),
            dino_pretrained=config.get("dino_pretrained", True),
            dino_frozen=config.get("dino_frozen", True),
            num_palette_experts=config.get("num_palette_experts", 3),
            use_palette_moe=config.get("use_palette_moe", True),
            semantic_gate_mode=config.get("semantic_gate_mode", "zero_scalar"),
            semantic_gate_init_bias=config.get("semantic_gate_init_bias", -3.0),
            semantic_gate_warmup_epochs=config.get("semantic_gate_warmup_epochs", 0),
            dino_trainable_blocks=config.get("dino_trainable_blocks", 0),
        )
    if config["model"] == "vietcolornet_v14":
        return VietColorNetV14(
            num_color_bins=config.get("num_color_bins", 259),
            temperature=config.get("color_temperature", 0.38),
            logits_stride=config.get("logits_stride", 2),
            backbone=config.get("backbone", "resnet34"),
            pretrained_encoder=config.get("pretrained_encoder", False),
            num_palette_queries=config.get("num_palette_queries", 32),
            max_chroma=config.get("max_chroma", 48.0),
            chroma_residual_scale=config.get("chroma_residual_scale", 0.7),
            use_global_chroma_scale=config.get("use_global_chroma_scale", True),
            dino_pretrained=config.get("dino_pretrained", True),
            dino_frozen=config.get("dino_frozen", True),
            num_palette_experts=config.get("num_palette_experts", 3),
            use_palette_moe=config.get("use_palette_moe", True),
            semantic_gate_mode=config.get("semantic_gate_mode", "zero_scalar"),
            semantic_gate_init_bias=config.get("semantic_gate_init_bias", -3.0),
            semantic_gate_warmup_epochs=config.get("semantic_gate_warmup_epochs", 0),
            dino_trainable_blocks=config.get("dino_trainable_blocks", 0),
        )
    if config["model"] == "ddcolor":
        from .ddcolor_adapter import DDColorAdapter
        return DDColorAdapter(config)
    raise ValueError(f"Unsupported model: {config['model']}")

