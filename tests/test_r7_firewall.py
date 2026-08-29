from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.r7_config import load_config, sha256_file
from src.r7_firewall import (
    DEVELOPMENT,
    FINAL_HOLDOUT,
    FirewallViolation,
    allowed_patient_ids,
    stream_stage_metadata,
    verify_partition,
)


class HoldoutFirewallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        project_root = Path(cls.config["_project_root"])
        manifest = project_root / cls.config["partition"]["manifest_path"]
        metadata = project_root / cls.config["source"]["metadata_path"]
        if not manifest.is_file() or not metadata.is_file():
            raise unittest.SkipTest(
                "credentialed BRSET metadata and the private patient partition "
                "are intentionally absent from the public repository"
            )

    def test_frozen_partition_integrity(self) -> None:
        integrity = verify_partition(self.config)
        self.assertEqual(len(integrity.development_ids), 5771)
        self.assertEqual(len(integrity.holdout_ids), 1924)
        self.assertFalse(integrity.development_ids & integrity.holdout_ids)
        self.assertEqual(
            integrity.manifest_sha256,
            "d8ac34c7a6303e39e8d18a761365490943d668aa781b552706b4fc2ce35d0ef5",
        )

    def test_holdout_denied_before_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = json.loads(json.dumps(self.config))
            config["firewall"]["holdout_release_marker"] = str(
                Path(directory) / "absent_release.json"
            )
            with self.assertRaises(FirewallViolation):
                allowed_patient_ids(
                    config, FINAL_HOLDOUT, purpose="prediction_generation"
                )

    def test_development_loader_materializes_only_development(self) -> None:
        records = stream_stage_metadata(
            self.config,
            DEVELOPMENT,
            columns=("image_id", "exam_eye", "drusen"),
            purpose="development_modeling",
        )
        development_ids = allowed_patient_ids(
            self.config, DEVELOPMENT, purpose="development_modeling"
        )
        self.assertEqual(len(records), 11542)
        self.assertEqual({r["patient_id"] for r in records}, set(development_ids))

    def test_tampered_manifest_is_rejected(self) -> None:
        original = Path(self.config["_project_root"]) / self.config["partition"][
            "manifest_path"
        ]
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "patient_partition.csv"
            tampered.write_bytes(original.read_bytes() + b"999999,development\n")
            config = json.loads(json.dumps(self.config))
            config["partition"]["manifest_path"] = str(tampered)
            with self.assertRaises(FirewallViolation):
                verify_partition(config)

    def test_raw_metadata_hash_is_stable(self) -> None:
        self.assertEqual(
            sha256_file(Path(self.config["source"]["metadata_path"])),
            self.config["source"]["metadata_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
