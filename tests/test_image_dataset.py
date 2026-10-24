import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "prepare"))

from colorization.image_dataset import (
    infer_collection_id,
    infer_content_category,
    has_near_duplicate,
    manipulation_reason,
)
from prepare_historical_image_protocol import split_by_collection


class ImageDatasetTests(unittest.TestCase):
    def test_manipulated_color_metadata_is_detected(self):
        self.assertEqual(manipulation_reason({"title": "Saigon colorized photograph"}), "colorized")
        self.assertIsNone(manipulation_reason({"title": "Saigon, 1968", "description": "Kodachrome"}))

    def test_content_category_uses_metadata(self):
        self.assertEqual(infer_content_category({"title": "Portrait of a soldier"}), "people")
        self.assertEqual(infer_content_category({"description": "Aircraft at an airfield"}), "transport")

    def test_collection_prefers_author(self):
        self.assertEqual(infer_collection_id({"author": "John Smith", "source_category": "Category:X"}),
                         "author-john-smith")

    def test_collection_split_has_no_leakage(self):
        rows = [{"collection_id": "a", "id": 1}, {"collection_id": "a", "id": 2},
                {"collection_id": "b", "id": 3}, {"collection_id": "c", "id": 4}]
        train, validation = split_by_collection(rows, 0.25, 2026)
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertFalse({row["collection_id"] for row in train}
                         & {row["collection_id"] for row in validation})

    def test_near_duplicate_hash(self):
        self.assertTrue(has_near_duplicate("0000000000000001", {"0000000000000000"}, 2))
        self.assertFalse(has_near_duplicate("000000000000000f", {"0000000000000000"}, 2))


if __name__ == "__main__":
    unittest.main()
