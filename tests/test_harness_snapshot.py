import tempfile
import unittest
from pathlib import Path

from ecdysis.harness_patch import revert_harness, snapshot_harness


class HarnessSnapshotTest(unittest.TestCase):
    def test_restores_preexisting_changes_and_removes_new_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            harness_dir = project_root / "tau2" / "harness"
            harness_dir.mkdir(parents=True)
            original = harness_dir / "airline.py"
            original.write_text("# user edit before experiment\n")
            nested = harness_dir / "support" / "rules.txt"
            nested.parent.mkdir()
            nested.write_text("original rules\n")

            snapshot = snapshot_harness(project_root=project_root)
            try:
                original.write_text("# evolution edit\n")
                nested.unlink()
                generated = harness_dir / "generated.py"
                generated.write_text("generated = True\n")

                revert_harness(snapshot, project_root=project_root)

                self.assertEqual(
                    original.read_text(), "# user edit before experiment\n"
                )
                self.assertEqual(nested.read_text(), "original rules\n")
                self.assertFalse(generated.exists())
            finally:
                snapshot.cleanup()
