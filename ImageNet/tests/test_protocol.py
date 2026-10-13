import json
import pickle
import unittest
from pathlib import Path

import numpy as np

from colorization.data import ImageNetDataset, MultiSourceVietnamDataset, VietnamDataset, sha256


ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_official_validation_partition(self):
        manifest_path = ROOT / "data" / "processed" / "protocol_manifest_v3.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        val = np.load(manifest_path.parent / manifest["splits"]["val"]["path"])
        test = np.load(manifest_path.parent / manifest["splits"]["test"]["path"])
        self.assertEqual(len(val), 5000)
        self.assertEqual(len(test), 45000)
        self.assertEqual(len(set(val.tolist()) & set(test.tolist())), 0)
        self.assertEqual(len(set(val.tolist()) | set(test.tolist())), 50000)

    def test_all_splits_load(self):
        manifest = ROOT / "data" / "processed" / "protocol_manifest_v3.json"
        for split in ("train", "val", "test"):
            dataset = ImageNetDataset(manifest, split, image_size=64, limit=1)
            self.assertEqual(tuple(dataset[0]["rgb"].shape), (3, 64, 64))

    def test_dataset_pickle_drops_open_memmaps(self):
        manifest = ROOT / "data" / "processed" / "protocol_manifest_v3.json"
        dataset = ImageNetDataset(manifest, "val", image_size=64, limit=2)
        expected_id = dataset[0]["id"]

        payload = pickle.dumps(dataset)
        restored = pickle.loads(payload)

        self.assertLess(len(payload), 100_000)
        self.assertIsNone(restored.rows)
        self.assertIsNone(restored.official_array)
        self.assertEqual(restored[0]["id"], expected_id)

    def test_vietnam_protocol_is_video_disjoint(self):
        manifest_path = ROOT / "data" / "vietnam_protocol_v1" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        video_sets = []
        count = 0
        for split in ("train", "val", "test"):
            info = manifest["splits"][split]
            csv_path = manifest_path.parent / info["path"]
            self.assertEqual(sha256(csv_path), info["sha256"])
            dataset = VietnamDataset(ROOT.parent / "dataset" / "VietNam_dataset", 64,
                                     csv_path=csv_path, split=split)
            self.assertEqual(len(dataset), info["count"])
            video_sets.append(set(info["videos"]))
            count += len(dataset)
        self.assertTrue(video_sets[0].isdisjoint(video_sets[1]))
        self.assertTrue(video_sets[0].isdisjoint(video_sets[2]))
        self.assertTrue(video_sets[1].isdisjoint(video_sets[2]))
        self.assertEqual(count, manifest["total_count"])

    def test_wikimedia_protocol_preserves_frozen_evaluation(self):
        v1_path = ROOT / "data" / "vietnam_protocol_v1" / "manifest.json"
        v2_path = ROOT / "data" / "vietnam_protocol_v2" / "manifest.json"
        v1 = json.loads(v1_path.read_text(encoding="utf-8"))
        v2 = json.loads(v2_path.read_text(encoding="utf-8"))
        self.assertEqual(v2["protocol"], "vietnam_video_disjoint_wikimedia_train_v2")
        self.assertEqual(v2["splits"]["train"]["source_counts"],
                         {"vietnam_video": 1235, "wikimedia_commons": 477})
        for split in ("val", "test"):
            self.assertEqual(v2["splits"][split]["sha256"], v1["splits"][split]["sha256"])
            self.assertEqual(sha256(v2_path.parent / v2["splits"][split]["path"]),
                             v1["splits"][split]["sha256"])

    def test_mixed_vietnam_training_sources_load(self):
        manifest_path = ROOT / "data" / "vietnam_protocol_v2" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        roots = {name: info["root"] for name, info in manifest["sources"].items()}
        dataset = MultiSourceVietnamDataset(
            roots, 64, csv_path=manifest_path.parent / manifest["splits"]["train"]["path"])
        source_indices = {}
        for index, row in enumerate(dataset.rows):
            source_indices.setdefault(row["source"], index)
        self.assertEqual(set(source_indices), {"vietnam_video", "wikimedia_commons"})
        for source, index in source_indices.items():
            sample = dataset[index]
            self.assertEqual(sample["source"], source)
            self.assertEqual(tuple(sample["rgb"].shape), (3, 64, 64))

    def test_historical_final_test_is_frozen_and_balanced(self):
        manifest_path = ROOT / "data" / "historical_vietnam_final_test_v1" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "FROZEN_DO_NOT_USE_FOR_TRAINING_OR_MODEL_SELECTION")
        test = manifest["test"]
        self.assertEqual(test["count"], 48)
        self.assertEqual(set(test["count_per_video"].values()), {16})
        csv_path = manifest_path.parent / test["path"]
        self.assertEqual(sha256(csv_path), test["sha256"])
        dataset = VietnamDataset(manifest_path.parent, 64, csv_path=csv_path, split="test")
        self.assertEqual(len(dataset), 48)
        self.assertEqual(tuple(dataset[0]["rgb"].shape), (3, 64, 64))
        for relative, expected in manifest["frame_sha256"].items():
            self.assertEqual(sha256(manifest_path.parent / relative), expected)

    def test_historical_adaptation_protocol_loads_all_sources(self):
        manifest_path = ROOT / "data" / "vietnam_protocol_v3" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["protocol"],
                         "vietnam_video_disjoint_wikimedia_historical_v3")
        self.assertFalse(manifest["frozen_historical_final_test_reused"])
        self.assertEqual(manifest["splits"]["train"]["source_counts"], {
            "historical_video": 447, "vietnam_video": 1235, "wikimedia_commons": 477})
        self.assertEqual(manifest["splits"]["val"]["source_counts"], {
            "historical_video": 145, "vietnam_video": 225})
        roots = {name: info["root"] for name, info in manifest["sources"].items()}
        for split, expected_sources in (("train", {"historical_video", "vietnam_video",
                                                     "wikimedia_commons"}),
                                        ("val", {"historical_video", "vietnam_video"})):
            info = manifest["splits"][split]
            csv_path = manifest_path.parent / info["path"]
            self.assertEqual(sha256(csv_path), info["sha256"])
            dataset = MultiSourceVietnamDataset(roots, 64, csv_path=csv_path)
            source_indices = {}
            for index, row in enumerate(dataset.rows):
                source_indices.setdefault(row["source"], index)
            self.assertEqual(set(source_indices), expected_sources)
            for source, index in source_indices.items():
                sample = dataset[index]
                self.assertEqual(sample["source"], source)
                self.assertEqual(tuple(sample["rgb"].shape), (3, 64, 64))


if __name__ == "__main__":
    unittest.main()
