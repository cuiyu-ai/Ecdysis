import json
import tempfile
import unittest
from pathlib import Path

from ecdysis.provenance import sanitize_for_persistence, write_run_manifest


class ProvenanceTest(unittest.TestCase):
    def test_recursive_redaction_preserves_env_names(self) -> None:
        payload = {
            "api_key": "secret-value",
            "api_key_env": "DASHSCOPE_API_KEY",
            "nested": [{"authorization": "Bearer secret"}],
        }

        sanitized = sanitize_for_persistence(payload)

        self.assertEqual(sanitized["api_key"], "<redacted>")
        self.assertEqual(sanitized["api_key_env"], "DASHSCOPE_API_KEY")
        self.assertEqual(sanitized["nested"][0]["authorization"], "<redacted>")

    def test_manifest_is_redacted_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = write_run_manifest(
                project_root=root,
                experiment="E5",
                experiment_config={"name": "debate", "api_key": "hidden"},
                evaluation_config={"domain": "airline"},
                config_path=root / "configs" / "E5.yaml",
                domain="airline",
                run_id="20260721_120000",
                dry_run=False,
                result_payload={"summary": {"pass@k": 0.5}},
            )

            payload = json.loads(manifest_path.read_text())

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["config"]["experiment"]["api_key"], "<redacted>")
        self.assertTrue(payload["config"]["sha256"])
        self.assertEqual(payload["result_summary"]["summary"]["pass@k"], 0.5)

