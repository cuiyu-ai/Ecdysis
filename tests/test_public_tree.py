from __future__ import annotations

import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRECTORIES = {
    "__pycache__",
    ".pytest_cache",
    "configs",
    "data",
    "logs",
    "simulations",
}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".yaml", ".yml"}
OLD_TERMINOLOGY = re.compile(
    "|".join(
        [
            r"\bReso" + r"nate\b",
            r"\bBo" + r"T\b",
            r"\bbo" + r"t\b",
        ]
    ),
    re.IGNORECASE,
)


class PublicTreeTest(unittest.TestCase):
    def test_excludes_configs_data_and_generated_files(self) -> None:
        violations: list[str] = []
        for path in PROJECT_ROOT.rglob("*"):
            relative = path.relative_to(PROJECT_ROOT)
            if relative.parts and relative.parts[0] == ".git":
                continue
            if any(part in FORBIDDEN_DIRECTORIES for part in relative.parts):
                violations.append(str(relative))
            elif path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
                violations.append(str(relative))

        self.assertEqual(violations, [])

    def test_source_uses_only_ecdysis_public_naming(self) -> None:
        violations: list[str] = []
        for path in PROJECT_ROOT.rglob("*"):
            relative = path.relative_to(PROJECT_ROOT)
            if relative.parts and relative.parts[0] == ".git":
                continue
            if path == Path(__file__):
                continue
            if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".toml"}:
                continue
            text = path.read_text(encoding="utf-8-sig")
            if OLD_TERMINOLOGY.search(text):
                violations.append(str(path.relative_to(PROJECT_ROOT)))

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
