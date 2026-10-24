import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "prepare"))

from collect_wikimedia_vietnam import (
    allowed_license,
    api_get,
    difference_hash,
    download_bytes,
    has_complete_metadata_cache,
    image_colorfulness,
    iter_downloads,
    round_robin,
)

class FakeResponse:
    def __init__(self, status_code=200, content=b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self.text = "fake response"

    def raise_for_status(self):
        pass

    def json(self):
        return {"query": {"pages": []}}


class FakeSession:
    api_delay = 0
    last_api_request = 0

    def __init__(self):
        self.get_calls = []
        self.post_calls = []
        self.get_responses = []

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        if self.get_responses:
            return self.get_responses.pop(0)
        return FakeResponse()

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return FakeResponse()


class WikimediaCollectorTests(unittest.TestCase):
    @patch("collect_wikimedia_vietnam.requests.Session")
    @patch("collect_wikimedia_vietnam.download_bytes")
    def test_parallel_downloads_preserve_order_and_skip_invalid_metadata(self, download, session):
        download.side_effect = lambda client, url, retries, timeout: url.encode()
        rows = [{"title": str(index), "mime": "image/jpeg", "width": 768,
                 "height": 768, "license": "CC BY 4.0", "download_url": f"url-{index}"}
                for index in range(4)]
        rows[1]["width"] = 100
        results = list(iter_downloads(rows, "test", 3, 25, 2, 512, False))
        self.assertEqual([row["title"] for row, content, error in results], ["0", "1", "2", "3"])
        self.assertEqual(results[0][1], b"url-0")
        self.assertIsNone(results[1][1])
        self.assertEqual(download.call_count, 3)

    def test_api_uses_post_for_long_metadata_queries(self):
        session = FakeSession()
        api_get(session, {"action": "query", "titles": "File:" + "x" * 9000}, method="post")

        self.assertEqual(len(session.get_calls), 0)
        self.assertEqual(len(session.post_calls), 1)
        self.assertIn("titles", session.post_calls[0][1]["data"])

    @patch("collect_wikimedia_vietnam.time.sleep")
    def test_download_retries_rate_limit(self, sleep):
        session = FakeSession()
        session.get_responses = [FakeResponse(429, headers={"Retry-After": "3"}),
                                 FakeResponse(200, content=b"image")]

        self.assertEqual(download_bytes(session, "https://example.test/image.jpg"), b"image")
        self.assertEqual(len(session.get_calls), 2)
        self.assertEqual(session.get_calls[0][1]["timeout"], (10, 25))
        sleep.assert_called_once_with(3.0)

    def test_license_allowlist(self):
        self.assertTrue(allowed_license("CC BY-SA 4.0"))
        self.assertTrue(allowed_license("Public domain"))
        self.assertTrue(allowed_license("KOGL Type 1"))
        self.assertFalse(allowed_license("All rights reserved"))
        self.assertFalse(allowed_license("CC BY-NC 4.0"))
        self.assertFalse(allowed_license("CC BY-ND 4.0"))
        self.assertFalse(allowed_license("KOGL Type 2"))
        self.assertFalse(allowed_license("KOGL Type 3"))
        self.assertFalse(allowed_license("KOGL Type 4"))

    def test_color_filter(self):
        gray = Image.new("RGB", (32, 32), (100, 100, 100))
        colorful = Image.new("RGB", (32, 32), (220, 30, 80))
        self.assertLess(image_colorfulness(gray), 1.0)
        self.assertGreater(image_colorfulness(colorful), 7.0)

    def test_difference_hash_is_stable(self):
        image = Image.new("RGB", (32, 32), (10, 20, 30))
        self.assertEqual(difference_hash(image), difference_hash(image.copy()))

    def test_round_robin_balances_categories(self):
        rows = [{"category": "a", "id": 1}, {"category": "a", "id": 2},
                {"category": "b", "id": 3}, {"category": "b", "id": 4}]
        self.assertEqual([row["category"] for row in round_robin(rows, "category")],
                         ["a", "b", "a", "b"])

    def test_completed_metadata_csv_enables_fast_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertFalse(has_complete_metadata_cache(output, [{"title": "File:A.jpg"}]))
            (output / "discovered_metadata.csv").write_text("title\nFile:A.jpg\n", encoding="utf-8")
            self.assertTrue(has_complete_metadata_cache(output, [{"title": "File:A.jpg"}]))


if __name__ == "__main__":
    unittest.main()
