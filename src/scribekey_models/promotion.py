from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from cryptography.hazmat.primitives.asymmetric import ec
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

DEFAULT_COMPATIBILITY: dict[str, Any] = {
    "minAndroidApiLevel": 28,
    "minAppVersionCode": 10,
    "catalogsSchemaVersion": 1,
    "supportedRuntimeFamilies": ["sherpa-onnx", "gguf", "pyannote"],
    "supportedConfigFamilies": [
        "speech-model-catalog",
        "cleanup-model-catalog",
        "speaker-diarization-manifest",
    ],
}


@dataclass(frozen=True)
class PromotionResult:
    channel: str
    release_id: str
    sequence: int
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
    speech_content: str,
    diarization_content: str,
    cleanup_content: str,
    git_commit: str | None = None,
    recovery_cleared_models: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

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
        "createdAt": now,
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
            "description": "Redistribution-cleared models explicitly permitted for mirror recovery",
            "clearedModels": recovery_cleared_models,
        }

    return manifest


def build_channel_distribution_manifest(
    channel: str,
    release_data: dict[str, Any],
    *,
    sequence: int,
    issued_at: str | None = None,
    compatibility: dict[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    rel_id = str(release_data["releaseId"])
    catalogs = release_data["catalogs"]
    now = issued_at or datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    comp = dict(compatibility) if compatibility else dict(DEFAULT_COMPATIBILITY)
    comp.setdefault("minAndroidApiLevel", 28)
    comp.setdefault("minAppVersionCode", 10)
    comp.setdefault("catalogsSchemaVersion", 1)
    comp.setdefault("supportedRuntimeFamilies", ["sherpa-onnx", "gguf", "pyannote"])
    comp.setdefault(
        "supportedConfigFamilies",
        ["speech-model-catalog", "cleanup-model-catalog", "speaker-diarization-manifest"],
    )

    manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "channel": channel,
        "sequence": sequence,
        "issuedAt": now,
        "releaseId": rel_id,
        "compatibility": comp,
        "catalogs": {
            "speech": {
                "filename": catalogs["speech"]["filename"],
                "path": f"../releases/{rel_id}/{catalogs['speech']['filename']}",
                "sha256": catalogs["speech"]["sha256"],
                "sizeBytes": catalogs["speech"]["sizeBytes"],
            },
            "diarization": {
                "filename": catalogs["diarization"]["filename"],
                "path": f"../releases/{rel_id}/{catalogs['diarization']['filename']}",
                "sha256": catalogs["diarization"]["sha256"],
                "sizeBytes": catalogs["diarization"]["sizeBytes"],
            },
            "cleanup": {
                "filename": catalogs["cleanup"]["filename"],
                "path": f"../releases/{rel_id}/{catalogs['cleanup']['filename']}",
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
    sequence: int | None = None,
    compatibility: dict[str, Any] | None = None,
    private_key: ec.EllipticCurvePrivateKey | None = None,
    key_id: str = DEFAULT_KEY_ID,
    root: Path = ROOT,
) -> PromotionResult:
    """Promote a channel to target release using staged validation for crash/failure safety."""
    if channel not in {"qa", "stable"}:
        raise ValueError(f"Invalid channel '{channel}': must be 'qa' or 'stable'")

    releases_dir = root / "catalog" / "releases"
    release_file = releases_dir / f"{release_id}.yaml"
    if not release_file.exists():
        raise FileNotFoundError(f"Release '{release_id}' definition does not exist at {release_file}")

    channels_file = root / "catalog" / "channels.yaml"
    if not channels_file.exists():
        raise FileNotFoundError(f"Channels file does not exist at {channels_file}")

    # 1. Load and validate release schema
    release_data = _load_yaml(release_file)
    rel_schema = _load_json(root / "schemas" / "release.schema.json")
    rel_validator = Draft202012Validator(rel_schema)
    rel_errors = list(rel_validator.iter_errors(release_data))
    if rel_errors:
        raise ValueError(f"Release '{release_id}' failed schema validation: {rel_errors[0].message}")

    # 2. Load existing channels
    channels_data = _load_yaml(channels_file)
    channel_info = channels_data.get("channels", {}).get(channel, {})
    current_release = str(channel_info.get("targetRelease", ""))
    current_sequence = int(channel_info.get("sequence", 0))

    channel_dist_dir = root / "generated" / "channels"
    channel_json_path = channel_dist_dir / f"{channel}.json"
    sig_path = channel_dist_dir / f"{channel}.json.sig"

    channel_comp = compatibility or channel_info.get("compatibility") or dict(DEFAULT_COMPATIBILITY)
    target_notes = notes or channel_info.get("notes") or release_data.get("description", "")

    # Check sequence monotonicity and anti-rollback requirements
    if sequence is not None:
        if sequence < current_sequence:
            raise ValueError(
                f"Promotion sequence must be strictly monotonically increasing: "
                f"proposed sequence {sequence} < current sequence {current_sequence}"
            )
        if sequence == current_sequence:
            # Same sequence is ONLY allowed when target release and metadata have not changed
            if current_release != release_id:
                raise ValueError(
                    f"Cannot modify target release to '{release_id}' at the same sequence {sequence}. "
                    "A changed publication requires a higher sequence."
                )
            if channel_json_path.exists():
                existing = _load_json(channel_json_path)
                if existing.get("notes") != target_notes:
                    raise ValueError(
                        f"Cannot modify notes at the same sequence {sequence}. "
                        "A changed publication requires a higher sequence."
                    )
                if existing.get("compatibility") != channel_comp:
                    raise ValueError(
                        f"Cannot modify compatibility at the same sequence {sequence}. "
                        "A changed publication requires a higher sequence."
                    )
            target_seq = sequence
        else:
            target_seq = sequence
    else:
        # sequence is None: check for idempotent re-run vs incrementing
        if current_release == release_id:
            # Check if any metadata would change at same sequence
            if channel_json_path.exists():
                existing = _load_json(channel_json_path)
                if existing.get("notes") != target_notes or existing.get("compatibility") != channel_comp:
                    # Metadata changed: requires higher sequence
                    target_seq = current_sequence + 1
                else:
                    target_seq = current_sequence if current_sequence > 0 else 1
            else:
                target_seq = current_sequence if current_sequence > 0 else 1
        else:
            target_seq = current_sequence + 1

    # Idempotent no-op check:
    # Same sequence + same release + existing files are valid -> do NOT change issuedAt or re-sign
    if (
        current_release == release_id
        and target_seq == current_sequence
        and channel_json_path.exists()
    ):
        existing_manifest = _load_json(channel_json_path)
        if (
            existing_manifest.get("releaseId") == release_id
            and existing_manifest.get("notes") == target_notes
            and existing_manifest.get("compatibility") == channel_comp
        ):
            # Check signature validity
            sig_valid = False
            if sig_path.exists():
                pub = private_key.public_key() if private_key else (root / "keys" / "release-signing.pub")
                sig_valid, _ = verify_file(channel_json_path, sig_path, public_key=pub)

            if sig_valid:
                return PromotionResult(
                    channel=channel,
                    release_id=release_id,
                    sequence=current_sequence,
                    updated=False,
                    manifest_path=channel_json_path,
                    signature_path=sig_path if sig_path.exists() else None,
                    message=(
                        f"Channel '{channel}' is already pointing at release '{release_id}' "
                        f"at sequence {current_sequence} with valid metadata/signature (idempotent no-op)"
                    ),
                )

    if private_key is None:
        raise ValueError("Promotion updates require a signing key")

    # 3. Perform generation, schema validation, and signing in TEMP STAGING DIRECTORY
    # This guarantees previous stable generated output is never corrupted or deleted on failure.
    now_iso = (
        channel_info.get("issuedAt")
        if target_seq == current_sequence and channel_info.get("issuedAt")
        else datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        staged_channel_json = temp_dir / f"{channel}.json"
        staged_sig_path = temp_dir / f"{channel}.json.sig"

        # Generate candidate distribution manifest
        dist_manifest = build_channel_distribution_manifest(
            channel,
            release_data,
            sequence=target_seq,
            issued_at=now_iso,
            compatibility=channel_comp,
            notes=target_notes,
        )
        staged_channel_json.write_text(_render_json(dist_manifest), encoding="utf-8")

        # Validate against channel distribution JSON schema
        dist_schema = _load_json(root / "schemas" / "channel_distribution.schema.json")
        dist_validator = Draft202012Validator(dist_schema)
        dist_errors = list(dist_validator.iter_errors(dist_manifest))
        if dist_errors:
            raise ValueError(f"Generated channel manifest failed schema validation: {dist_errors[0].message}")

        # Sign candidate manifest in staging if private key provided
        has_sig = False
        if private_key is not None:
            sign_file(staged_channel_json, private_key, key_id=key_id, sig_path=staged_sig_path)
            # Verify newly created signature in staging
            pub = private_key.public_key()
            ok, msg = verify_file(staged_channel_json, staged_sig_path, public_key=pub)
            if not ok:
                raise RuntimeError(f"Signature verification failed immediately after signing: {msg}")
            has_sig = True

        # Prepare updated channels.yaml
        new_channels_data = {
            "schema_version": 1,
            "channels": dict(channels_data.get("channels", {})),
        }
        new_channel_entry: dict[str, Any] = {
            "targetRelease": release_id,
            "sequence": target_seq,
            "issuedAt": now_iso,
            "notes": target_notes,
            "compatibility": channel_comp,
        }
        new_channels_data["channels"][channel] = new_channel_entry

        # Validate updated channels.yaml schema
        chan_schema = _load_json(root / "schemas" / "channels.schema.json")
        chan_validator = Draft202012Validator(chan_schema)
        chan_errors = list(chan_validator.iter_errors(new_channels_data))
        if chan_errors:
            raise ValueError(f"Updated channels.yaml failed schema validation: {chan_errors[0].message}")

        # 4. ATOMIC COMMIT (only reached when all staging validation and signing succeeded)
        channel_dist_dir.mkdir(parents=True, exist_ok=True)

        # Atomic write to channels.yaml
        tmp_channels_file = channels_file.with_name(f".{channels_file.name}.tmp")
        tmp_channels_file.write_text(yaml.safe_dump(new_channels_data, sort_keys=False), encoding="utf-8")
        os.replace(tmp_channels_file, channels_file)

        # Atomic write to generated/channels/{channel}.json
        tmp_json = channel_dist_dir / f".{channel}.json.tmp"
        shutil.copyfile(staged_channel_json, tmp_json)
        os.replace(tmp_json, channel_json_path)

        out_sig_path: Path | None = None
        if has_sig:
            tmp_sig = channel_dist_dir / f".{channel}.json.sig.tmp"
            shutil.copyfile(staged_sig_path, tmp_sig)
            os.replace(tmp_sig, sig_path)
            out_sig_path = sig_path

        return PromotionResult(
            channel=channel,
            release_id=release_id,
            sequence=target_seq,
            updated=True,
            manifest_path=channel_json_path,
            signature_path=out_sig_path,
            message=f"Successfully promoted channel '{channel}' to release '{release_id}' at sequence {target_seq}",
        )
