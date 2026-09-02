"""Safe run provenance and result redaction helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "authorization",
    "password",
    "secret",
    "access_token",
    "refresh_token",
)


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower()
    return not normalized.endswith("_env") and any(
        fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS
    )


def sanitize_for_persistence(value: Any) -> Any:
    """Return a recursively redacted copy suitable for JSON persistence."""
    if isinstance(value, dict):
        return {
            str(key): (
                "<redacted>"
                if _is_sensitive_key(str(key))
                else sanitize_for_persistence(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_for_persistence(item) for item in value]
    return value


def sanitize_json_file(path: Path) -> bool:
    """Redact credential-like fields in a newly created JSON result file."""
    if not path.exists():
        return False
    try:
        original = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    sanitized = sanitize_for_persistence(original)
    if sanitized == original:
        return False
    path.write_text(json.dumps(sanitized, indent=2))
    return True


def _git_output(project_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _config_sha256(payload: Any) -> str:
    canonical = json.dumps(
        sanitize_for_persistence(payload),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_run_manifest(
    *,
    project_root: Path,
    experiment: str,
    experiment_config: dict[str, Any],
    evaluation_config: dict[str, Any],
    config_path: Path,
    domain: str,
    run_id: str,
    dry_run: bool,
    result_payload: dict[str, Any],
) -> Path:
    """Persist redacted provenance for one newly completed experiment."""
    slug = f"{experiment}_{experiment_config.get('name', experiment.lower())}"
    out_dir = project_root / "data" / "run_manifests" / slug / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    status = _git_output(project_root, "status", "--porcelain") or ""
    diff = _git_output(project_root, "diff", "--binary") or ""
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "domain": domain,
        "dry_run": dry_run,
        "config": {
            "path": str(config_path),
            "experiment": experiment_config,
            "evaluation": evaluation_config,
            "sha256": _config_sha256(
                {"experiment": experiment_config, "evaluation": evaluation_config}
            ),
        },
        "code": {
            "commit": _git_output(project_root, "rev-parse", "HEAD"),
            "branch": _git_output(project_root, "branch", "--show-current"),
            "dirty": bool(status),
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "result_summary": result_payload,
    }
    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(sanitize_for_persistence(manifest), indent=2))
    return out_path

