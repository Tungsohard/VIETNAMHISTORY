import io
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

import server


class FakeEngine:
    segmenter = None
    device_name = "test"

    def colorize_pil(self, image, strength=1.0, mode=None):
        return image.convert("RGB")


def png(size=(40, 30), mode="RGB"):
    buffer = io.BytesIO()
    Image.new(mode, size, 120).save(buffer, "PNG")
    return buffer.getvalue()


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        engines = {"vietcolornet": {"engine": FakeEngine(), "label": "VietColorNet"},
                   "ddcolor": {"engine": FakeEngine(), "label": "DDColor"}}
        self.client = TestClient(server.create_app(engines, Path(self.folder.name)))

    def tearDown(self):
        self.folder.cleanup()

    def colorize(self, data=None, **fields):
        form = {"mode": "natural", "strength": "1.0", "model": "", **fields}
        return self.client.post("/api/colorize", files={"file": ("x.png", png() if data is None else data, "image/png")}, data=form)

    def test_valid_upload(self):
        reply = self.colorize()
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(reply.json()["strength"], 1.0)

    def test_strength_must_be_finite_and_in_range(self):
        for value in ("nan", "inf", "-1", "0.1", "3"):
            with self.subTest(strength=value):
                self.assertEqual(self.colorize(strength=value).status_code, 400)

    def test_strength_limits_are_accepted(self):
        for value in ("0.5", "1.5"):
            with self.subTest(strength=value):
                self.assertEqual(self.colorize(strength=value).status_code, 200)

    def test_unknown_model_is_rejected(self):
        self.assertEqual(self.colorize(model="no_such_model").status_code, 400)
        self.assertEqual(self.colorize(model="ddcolor").status_code, 200)

    def test_invalid_mode_and_bad_files(self):
        self.assertEqual(self.colorize(mode="hack").status_code, 400)
        self.assertEqual(self.colorize(data=b"not an image").status_code, 400)
        self.assertEqual(self.colorize(data=b"").status_code, 400)

    def test_pixel_limit_checked_from_the_header(self):
        old = server.MAX_PIXELS
        server.MAX_PIXELS = 1000          # 40 x 30 = 1200 pixels is now too large
        try:
            reply = self.colorize()
        finally:
            server.MAX_PIXELS = old
        self.assertEqual(reply.status_code, 400)
        self.assertIn("megapixel", reply.json()["detail"])

    def test_api_schema_is_not_published(self):
        for path in ("/openapi.json", "/docs", "/redoc"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)

    def test_results_cannot_leave_the_output_folder(self):
        for path in ("/results/../README.md", "/results/..%2fREADME.md", "/results/..%5cREADME.md"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)


if __name__ == "__main__":
    unittest.main()
