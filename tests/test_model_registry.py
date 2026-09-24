from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ModelRegistryTests(unittest.TestCase):
    def test_registered_artifacts_and_documentation_are_consistent(self) -> None:
        registry = json.loads((ROOT / "models" / "registry.json").read_text())

        self.assertEqual(1, registry["schema_version"])
        models = registry["models"]
        self.assertTrue(models)
        self.assertIn(registry["default_model_id"], {item["model_id"] for item in models})

        for item in models:
            artifact_dir = ROOT / item["artifact_directory"]
            model_path = artifact_dir / "model.tflite"
            metadata_path = artifact_dir / "model-metadata.json"
            documentation_path = ROOT / item["documentation"]
            notebook_path = ROOT / item["training_notebook"]
            metadata = json.loads(metadata_path.read_text())
            checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()

            self.assertTrue(documentation_path.is_file())
            self.assertTrue(notebook_path.is_file())
            self.assertEqual(item["model_id"], metadata["model_id"])
            self.assertEqual(item["model_sha256"], checksum)
            self.assertEqual(metadata["model_sha256"], checksum)
            self.assertEqual(item["artifact_role"], metadata["artifact_role"])

    def test_only_a_deployable_model_can_be_the_default(self) -> None:
        registry = json.loads((ROOT / "models" / "registry.json").read_text())
        default = next(
            item for item in registry["models"]
            if item["model_id"] == registry["default_model_id"]
        )

        self.assertIn(default["status"], {"development_default", "pi_validated"})


if __name__ == "__main__":
    unittest.main()
