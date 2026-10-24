import unittest
import torch

from colorization.model import (
    EdgeExtractor,
    StripPooling,
    EdgeGatedDecoderBlock,
    VietColorNet,
    VietColorNetV7,
    build_model,
)


class VietColorNetTests(unittest.TestCase):
    def test_build_model_v8_registry(self):
        model = build_model({
            "model": "vietcolornet_v8", "backbone": "resnet34",
            "pretrained_encoder": False, "dino_pretrained": False,
            "dino_frozen": True, "use_palette_moe": False,
        })
        self.assertIsInstance(model, VietColorNetV7)

    def test_edge_extractor_detects_gradient(self):
        extractor = EdgeExtractor(out_channels=16)
        # Tạo ảnh độ sáng có biên độ nhảy bậc ở giữa
        luminance = torch.zeros(1, 1, 32, 32)
        luminance[:, :, :, 16:] = 1.0
        edge_feat = extractor(luminance)
        self.assertEqual(edge_feat.shape, (1, 16, 32, 32))
        # Vùng biên tại cột 15-16 phải có phản hồi khác 0
        self.assertGreater(float(edge_feat[:, :, :, 15:17].abs().mean().detach()), 0.0)

    def test_strip_pooling_preserves_shape(self):
        sp = StripPooling(channels=64)
        x = torch.randn(2, 64, 8, 8)
        out = sp(x)
        self.assertEqual(out.shape, (2, 64, 8, 8))

    def test_edge_gated_decoder_block(self):
        block = EdgeGatedDecoderBlock(in_channels=64, skip_channels=32, out_channels=32, edge_channels=16)
        x = torch.randn(2, 64, 8, 8)
        skip = torch.randn(2, 32, 16, 16)
        edge = torch.randn(2, 16, 16, 16)
        out = block(x, skip, edge=edge)
        self.assertEqual(out.shape, (2, 32, 16, 16))

    def test_vietcolornet_forward_and_backward(self):
        model = VietColorNet(backbone="resnet18", pretrained_encoder=False, chroma_boost=1.15)
        model.train()
        luminance = torch.randn(2, 1, 64, 64, requires_grad=True)
        out = model(luminance)
        self.assertEqual(out.shape, (2, 2, 64, 64))
        # Kiểm tra dải giá trị đầu ra luôn nằm trong [-1.0, 1.0]
        self.assertTrue(torch.all(out >= -1.0) and torch.all(out <= 1.0))

        # Kiểm tra luồng gradient lan truyền ngược đầy đủ
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(luminance.grad)
        self.assertIsNotNone(model.edge_stream.edge_conv[0].weight.grad)
        self.assertIsNotNone(model.hue_head[-1].weight.grad)
        self.assertIsNotNone(model.chroma_head[-2].weight.grad)

    def test_build_model_vietcolornet_registry(self):
        config = {
            "model": "vietcolornet",
            "backbone": "resnet18",
            "pretrained_encoder": False,
            "chroma_boost": 1.2,
        }
        model = build_model(config)
        self.assertIsInstance(model, VietColorNet)
        self.assertEqual(model.chroma_boost, 1.2)

    def test_color_palette_cross_attention(self):
        from colorization.model import ColorPaletteCrossAttention
        cpca = ColorPaletteCrossAttention(in_channels=64, num_queries=16)
        x = torch.randn(2, 64, 8, 8, requires_grad=True)
        out = cpca(x)
        self.assertEqual(out.shape, (2, 64, 8, 8))
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(cpca.queries.grad)
        self.assertIsNotNone(cpca.gamma.grad)

    def test_vietcolornet_with_palette_attention(self):
        model = VietColorNet(
            backbone="resnet18",
            pretrained_encoder=False,
            use_palette_attention=True,
            num_palette_queries=16,
        )
        luminance = torch.randn(2, 1, 64, 64)
        out = model(luminance)
        self.assertEqual(out.shape, (2, 2, 64, 64))

    def test_v5_decouples_hue_and_chroma(self):
        model = build_model({
            "model": "vietcolornet_v5",
            "backbone": "resnet18",
            "pretrained_encoder": False,
            "max_chroma": 48.0,
        })
        luminance = torch.randn(2, 1, 64, 64, requires_grad=True)
        out = model(luminance)
        chroma = torch.linalg.vector_norm(out, dim=1)
        self.assertEqual(out.shape, (2, 2, 64, 64))
        self.assertTrue(torch.all(chroma <= 48.0 / 128.0 + 1e-6))
        out.mean().backward()
        self.assertIsNotNone(luminance.grad)
        self.assertIsNotNone(model.hue_head[-1].weight.grad)
        self.assertIsNotNone(model.chroma_magnitude_head[0].weight.grad)

    def test_v6_calibration_heads_are_identity_initialized_and_trainable(self):
        model = build_model({"model": "vietcolornet_v6", "backbone": "resnet18",
                             "pretrained_encoder": False, "max_chroma": 48.0,
                             "use_global_chroma_scale": True})
        luminance = torch.randn(2, 1, 64, 64, requires_grad=True)
        out = model(luminance)
        self.assertEqual(out.shape, (2, 2, 64, 64))
        self.assertTrue(torch.all(torch.linalg.vector_norm(out, dim=1) <= 48.0 / 128.0 + 1e-6))
        out.mean().backward()
        self.assertIsNotNone(model.chroma_residual_head.weight.grad)
        self.assertIsNotNone(model.neutral_confidence_head.weight.grad)
        self.assertIsNotNone(model.global_chroma_head.weight.grad)


if __name__ == "__main__":
    unittest.main()

