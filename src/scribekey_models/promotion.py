from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from cryptography.hazmat.primitives.asymmetric import ed25519
from jsonschema import Draft202012Validator

from scribekey_models.signing import (
    DEFAULT_KEY_ID,
    sign_file,
    verify_file,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOG_DIR = ROOT / "catalog"
RELEASES_DIR = CATALOG_DIR / "releases"
CHANNELS_FILE = CATALOG_DIR / "channels.yaml"
GENERATED_DIR = ROOT / "generated"
SCHEMA_DIR = ROOT / "schemas"


@dataclass(frozen=True)
class PromotionResult:
    channel: str
    release_id: str
    updated: bool
    manifest_path: Path
    signature_path: Path | None
    message: str


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping at {path}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return data


def _render_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def create_release_manifest(
    release_id: str,
    description: str,
    *,
    created_at: str,
    speech_content: str,
    diarization_content: str,
    cleanup_content: str,
    git_commit: str | None = None,
    recovery_cleared_models: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    def _file_info(filename: str, content: str) -> dict[str, Any]:
        data_bytes = content.encode("utf-8")
        return {
            "filename": filename,
            "sha256": hashlib.sha256(data_bytes).hexdigest(),
            "sizeBytes": len(data_bytes),
        }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "releaseId": release_id,
        "createdAt": created_at,
        "description": description,
        "catalogs": {
            "speech": _file_info("model_catalog.json", speech_content),
            "diarization": _file_info("speaker_diarization_manifest.json", diarization_content),
            "cleanup": _file_info("cleanup_model_catalog.json", cleanup_content),
        },
    }

    if git_commit:
        manifest["gitCommit"] = git_commit

    if recovery_cleared_models:
        manifest["recoveryAssets"] = {
            "description": "Redistribution-cleared models available for mirror recovery",
            "clearedModels": recovery_cleared_models,
        }

    return manifest


def build_channel_distribution_manifest(
    channel: str,
    release_data: dict[str, Any],
    *,
    updated_at: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    rel_id = release_data["releaseId"]
    catalogs = release_data["catalogs"]
    deterministic_updated_at = updated_at or str(release_data["createdAt"])

    manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "channel": channel,
        "releaseId": rel_id,
        "updatedAt": deterministic_updated_at,
        "catalogs": {
            "speech": {
                "filename": catalogs["speech"]["filename"],
                "path": f"releases/{rel_id}/{catalogs['speech']['filename']}",
                "sha256": catalogs["speech"]["sha256"],
                "sizeBytes": catalogs["speech"]["sizeBytes"],
            },
            "diarization": {
                "filename": catalogs["diarization"]["filename"],
                "path": f"releases/{rel_id}/{catalogs['diarization']['filename']}",
                "sha256": catalogs["diarization"]["sha256"],
                "sizeBytes": catalogs["diarization"]["sizeBytes"],
            },
            "cleanup": {
                "filename": catalogs["cleanup"]["filename"],
                "path": f"releases/{rel_id}/{catalogs['cleanup']['filename']}",
                "sha256": catalogs["cleanup"]["sha256"],
                "sizeBytes": catalogs["cleanup"]["sizeBytes"],
            },
        },
    }

    if notes or release_data.get("description"):
        manifest["notes"] = notes or release_data.get("description")

    if release_data.get("gitCommit"):
        manifest["gitCommit"] = release_data["gitCommit"]

    if release_data.get("recoveryAssets"):
        manifest["recoveryAssets"] = release_data["recoveryAssets"]

    return manifest


def promote(
    channel: str,
    release_id: str,
    *,
    notes: str = "",
    private_key: ed25519.Ed25519PrivateKey,
    key_id: str = DEFAULT_KEY_ID,
    root: Path = ROOT,
) -> PromotionResult:
    if channel not in {"qa", "stable"}:
        raise ValueError(f"Invalid channel '{channel}': must be 'qa' or 'stable'")

    releases_dir = root / "catalog" / "releases"
    release_file = releases_dir / f"{release_id}.yaml"
    if not release_file.exists():
        raise FileNotFoundError(f"Release '{release_id}' definition does not exist at {release_file}")

    channels_file = root / "catalog" / "channels.yaml"
    if not channels_file.exists():
        raise FileNotFoundError(f"Channels file does not exist at {channels_file}")

    # Load and validate release schema
    release_data = _load_yaml(release_file)
    rel_schema = _load_json(root / "schemas" / "release.schema.json")
    rel_validator = Draft202012Validator(rel_schema)
    rel_errors = list(rel_validator.iter_errors(release_data))
    if rel_errors:
        raise ValueError(f"Release '{release_id}' failed schema validation: {rel_errors[0].message}")

    # Load existing channels
    channels_data = _load_yaml(channels_file)
    channel_info = channels_data.get("channels", {}).get(channel, {})
    current_release = channel_info.get("targetRelease")

    # Target files
    channel_dist_dir = root / "generated" / "channels"
    channel_dist_dir.mkdir(parents=True, exist_ok=True)
    channel_json_path = channel_dist_dir / f"{channel}.json"
    sig_path = channel_dist_dir / f"{channel}.json.sig"

    # Backup for failure-safety (atomic rollback)
    old_channels_yaml = channels_file.read_text(encoding="utf-8")
    old_channel_json = channel_json_path.read_text(encoding="utf-8") if channel_json_path.exists() else None
    old_sig = sig_path.read_text(encoding="utf-8") if sig_path.exists() else None

    try:
        deterministic_updated_at = str(release_data["createdAt"])
        effective_notes = notes or str(release_data.get("description", ""))
        expected_channel_entry: dict[str, Any] = {
            "targetRelease": release_id,
            "updatedAt": deterministic_updated_at,
        }
        if effective_notes:
            expected_channel_entry["notes"] = effective_notes
        expected_manifest = build_channel_distribution_manifest(
            channel,
            release_data,
            updated_at=deterministic_updated_at,
            notes=effective_notes or None,
        )
        expected_json = _render_json(expected_manifest)

        # Same logical promotion produces exactly the same bytes and signature.
        if (
            current_release == release_id
            and channel_info == expected_channel_entry
            and channel_json_path.exists()
            and channel_json_path.read_text(encoding="utf-8") == expected_json
            and sig_path.exists()
            and verify_file(
                channel_json_path,
                sig_path,
                public_key=private_key.public_key(),
            )[0]
        ):
            return PromotionResult(
                channel=channel,
                release_id=release_id,
                updated=False,
                manifest_path=channel_json_path,
                signature_path=sig_path if sig_path.exists() else None,
                message=f"Channel '{channel}' is already pointing at release '{release_id}' (idempotent)",
            )

        # Update the pointer with deterministic release-derived metadata.
        channels_data["channels"][channel] = expected_channel_entry
        channels_file.write_text(yaml.safe_dump(channels_data, sort_keys=False), encoding="utf-8")

        channel_json_path.write_text(expected_json, encoding="utf-8")

        # Validate channel distribution manifest schema
        dist_schema = _load_json(root / "schemas" / "channel_distribution.schema.json")
        dist_validator = Draft202012Validator(dist_schema)
        dist_errors = list(dist_validator.iter_errors(expected_manifest))
        if dist_errors:
            raise ValueError(f"Generated channel manifest failed schema validation: {dist_errors[0].message}")

        out_sig_path = sign_file(channel_json_path, private_key, key_id=key_id, sig_path=sig_path)
        ok, msg = verify_file(
            channel_json_path,
            out_sig_path,
            public_key=private_key.public_key(),
        )
        if not ok:
            raise RuntimeError(f"Signature verification failed immediately after signing: {msg}")

        return PromotionResult(
            channel=channel,
            release_id=release_id,
            updated=True,
            manifest_path=channel_json_path,
            signature_path=out_sig_path,
            message=f"Successfully promoted channel '{channel}' to release '{release_id}'",
        )

    except Exception:
        # Atomic rollback on failure
        channels_file.write_text(old_channels_yaml, encoding="utf-8")
        if old_channel_json is not None:
            channel_json_path.write_text(old_channel_json, encoding="utf-8")
        elif channel_json_path.exists():
            channel_json_path.unlink()
        if old_sig is not None:
            sig_path.write_text(old_sig, encoding="utf-8")
        elif sig_path.exists():
            sig_path.unlink()
        raise
