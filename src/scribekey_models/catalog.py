from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from scribekey_models.guardrails import (
    validate_channel_and_release_pointers,
    validate_identities_and_integrity,
    validate_immutable_source_refs,
    validate_no_executable_payloads,
    validate_redistribution_clearance,
    validate_release_safety,
    validate_release_snapshot_integrity,
    validate_signatures,
)
from scribekey_models.promotion import build_channel_distribution_manifest

ROOT = Path(__file__).resolve().parents[2]
CATALOG_DIR = ROOT / "catalog"
RELEASES_DIR = CATALOG_DIR / "releases"
CHANNELS_FILE = CATALOG_DIR / "channels.yaml"
SCHEMA_DIR = ROOT / "schemas"
GENERATED_DIR = ROOT / "generated"


@dataclass(frozen=True)
class ValidationIssue:
    source: str
    message: str


@dataclass(frozen=True)
class GenerationArtifact:
    filename: str
    content: str


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


def load_speech_catalog_data() -> dict[str, Any]:
    return _load_yaml(CATALOG_DIR / "speech.yaml")


def load_diarization_catalog_data() -> dict[str, Any]:
    return _load_yaml(CATALOG_DIR / "diarization.yaml")


def load_cleanup_catalog_data() -> dict[str, Any]:
    return _load_yaml(CATALOG_DIR / "cleanup.yaml")


def load_channels_data() -> dict[str, Any]:
    return _load_yaml(CHANNELS_FILE)


def load_all_releases_data() -> dict[str, dict[str, Any]]:
    releases: dict[str, dict[str, Any]] = {}
    if not RELEASES_DIR.exists():
        return releases
    for rel_file in sorted(RELEASES_DIR.glob("*.yaml")):
        data = _load_yaml(rel_file)
        # Ensure releaseId matches filename stem
        rel_id = str(data.get("releaseId", rel_file.stem))
        releases[rel_id] = data
    return releases


def generate_speech_catalog() -> dict[str, Any]:
    raw = load_speech_catalog_data()
    models: list[dict[str, Any]] = []
    for model in raw.get("models", []):
        entry: dict[str, Any] = {"id": model["id"]}
        if model.get("outcome") is not None:
            entry["outcome"] = model["outcome"]
        entry.update(
            {
                "family": model["family"],
                "displayName": model["displayName"],
                "badge": model.get("badge", ""),
                "diskMb": model["diskMb"],
                "minRamMb": model["minRamMb"],
                "minApiLevel": model["minApiLevel"],
                "transcriptionMode": model.get("transcriptionMode", "SEGMENTED"),
                "supportsTimestamps": model.get("supportsTimestamps", False),
                "supportsPunctuation": model.get("supportsPunctuation", False),
                "supportsLanguageDetection": model.get("supportsLanguageDetection", False),
            }
        )
        if model.get("retired"):
            entry["retired"] = True
        if model.get("deprecated"):
            entry["deprecated"] = True
        if model.get("replacementId"):
            entry["replacementId"] = model["replacementId"]
        if model.get("streamLanguage"):
            entry["streamLanguage"] = model["streamLanguage"]
        entry["files"] = [
            {
                "name": file["name"],
                "downloadUrl": file["downloadUrl"],
                "sizeBytes": file["sizeBytes"],
                "sha256": file["sha256"],
            }
            for file in model["files"]
        ]
        entry["sherpaConfig"] = {
            "type": model["sherpaConfig"]["type"],
            "modelDir": model["sherpaConfig"].get("modelDir", ""),
        }
        entry["info"] = {
            "paramsBadge": model["info"]["paramsBadge"],
            "architecture": model["info"]["architecture"],
            "languages": model["info"]["languages"],
        }
        if model.get("languageTiers"):
            entry["languageTiers"] = model["languageTiers"]
        if model.get("provenance"):
            entry["provenance"] = model["provenance"]
        models.append(entry)
    return {"models": models}


def generate_diarization_manifest() -> dict[str, Any]:
    raw = load_diarization_catalog_data()
    return {
        "packageId": raw["packageId"],
        "runtime": {
            "name": raw["runtime"]["name"],
            "version": str(raw["runtime"]["version"]),
            "revision": raw["runtime"]["revision"],
        },
        "models": [
            {
                "role": model["role"],
                "fileName": model["fileName"],
                "downloadUrl": model["downloadUrl"],
                "sizeBytes": model["sizeBytes"],
                "sha256": model["sha256"],
                "sourceRepository": model["sourceRepository"],
                "sourceRevision": model["sourceRevision"],
                "licenseId": model["licenseId"],
                "licenseUrl": model["licenseUrl"],
            }
            for model in raw["models"]
        ],
    }


def _cleanup_entry(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "modelId": model["modelId"],
        "displayName": model["displayName"],
        "revision": model["revision"],
        "bundleFileName": model["bundleFileName"],
        "downloadUrl": model["downloadUrl"],
        "sizeBytes": model["sizeBytes"],
        "sha256": model["sha256"],
        "minDeviceMemoryGb": model["minDeviceMemoryGb"],
        "minimumApiLevel": model.get("minimumApiLevel", 28),
        "supportedAbis": model.get("supportedAbis", ["arm64-v8a", "x86_64"]),
        "supportedLanguages": model.get("supportedLanguages", ["en"]),
        "systemPrompt": model["systemPrompt"],
        "promptTemplate": model["promptTemplate"],
        "maxInputCharacters": model["maxInputCharacters"],
        "maxOutputCharacters": model["maxOutputCharacters"],
        "maxOutputTokens": model["maxOutputTokens"],
        "contextTokens": model["contextTokens"],
        "topK": model.get("topK", 1),
        "topP": float(model.get("topP", 1.0)),
        "temperature": float(model.get("temperature", 0.0)),
        "repetitionPenalty": (
            float(model["repetitionPenalty"])
            if model.get("repetitionPenalty") is not None
            else None
        ),
        "deterministicDecoding": model.get("deterministicDecoding", True),
        "samplerSeed": model.get("samplerSeed", 42),
        "license": model["license"],
        "attribution": model["attribution"],
        "licenseUrl": model["licenseUrl"],
        "privacyDisclosure": model.get(
            "privacyDisclosure", "Transcript text stays on this device."
        ),
    }


def generate_cleanup_catalog() -> dict[str, Any]:
    raw = load_cleanup_catalog_data()
    production = raw.get("production")
    return {
        "production": _cleanup_entry(production) if production else None,
        "candidates": [_cleanup_entry(candidate) for candidate in raw.get("candidates", [])],
    }


def _render_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def generate_all_artifacts() -> list[GenerationArtifact]:
    artifacts: list[GenerationArtifact] = []

    speech_json = _render_json(generate_speech_catalog())
    diarization_json = _render_json(generate_diarization_manifest())
    cleanup_json = _render_json(generate_cleanup_catalog())

    # Root runtime catalogues
    artifacts.append(GenerationArtifact("model_catalog.json", speech_json))
    artifacts.append(GenerationArtifact("speaker_diarization_manifest.json", diarization_json))
    artifacts.append(GenerationArtifact("cleanup_model_catalog.json", cleanup_json))

    # Release manifests are derived metadata. The catalogue files below each release are
    # deliberately NOT generated here: `release create` freezes them once and validation checks
    # their declared hashes. Re-generating them from the live catalogue would mutate history.
    releases = load_all_releases_data()
    for rel_id, rel_data in releases.items():
        rel_dir = f"releases/{rel_id}"
        artifacts.append(GenerationArtifact(f"{rel_dir}/release.json", _render_json(rel_data)))

    # Channel distribution manifests
    if CHANNELS_FILE.exists():
        channels_data = load_channels_data()
        for chan_name in ["qa", "stable"]:
            chan_info = channels_data.get("channels", {}).get(chan_name)
            if chan_info:
                rel_id = str(chan_info["targetRelease"])
                if rel_id in releases:
                    manifest = build_channel_distribution_manifest(
                        chan_name,
                        releases[rel_id],
                        sequence=int(chan_info.get("sequence", 1)),
                        issued_at=chan_info.get("issuedAt"),
                        compatibility=chan_info.get("compatibility"),
                        notes=chan_info.get("notes"),
                    )
                    artifacts.append(
                        GenerationArtifact(f"channels/{chan_name}.json", _render_json(manifest))
                    )

    return artifacts


def export_generated(output_dir: Path = GENERATED_DIR, *, check: bool = False) -> list[str]:
    drift: list[str] = []
    for artifact in generate_all_artifacts():
        target = output_dir / artifact.filename
        if check:
            if not target.exists():
                drift.append(f"Missing generated file: {target}")
            elif target.read_text(encoding="utf-8") != artifact.content:
                drift.append(f"Generated file is stale: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(artifact.content, encoding="utf-8")
    return drift


def validate() -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    # 1. Validate canonical YAML sources with JSON schemas
    canonical_sources = [
        (CATALOG_DIR / "speech.yaml", SCHEMA_DIR / "speech.schema.json"),
        (CATALOG_DIR / "cleanup.yaml", SCHEMA_DIR / "cleanup.schema.json"),
        (CATALOG_DIR / "diarization.yaml", SCHEMA_DIR / "diarization.schema.json"),
    ]
    if CHANNELS_FILE.exists():
        canonical_sources.append((CHANNELS_FILE, SCHEMA_DIR / "channels.schema.json"))

    if RELEASES_DIR.exists():
        for rel_file in RELEASES_DIR.glob("*.yaml"):
            canonical_sources.append((rel_file, SCHEMA_DIR / "release.schema.json"))

    for source, schema in canonical_sources:
        if not source.exists():
            continue
        try:
            validator = Draft202012Validator(_load_json(schema))
            for error in validator.iter_errors(_load_yaml(source)):
                location = ".".join(str(part) for part in error.absolute_path)
                prefix = f"{location}: " if location else ""
                issues.append(ValidationIssue(str(source.relative_to(ROOT)), prefix + error.message))
        except (OSError, TypeError, ValueError, yaml.YAMLError, SchemaError) as exc:
            issues.append(ValidationIssue(str(source.relative_to(ROOT)), str(exc)))

    # 2. Guardrails: source refs, identities, safety, pointers, redistribution
    speech_data = load_speech_catalog_data()
    cleanup_data = load_cleanup_catalog_data()
    diarization_data = load_diarization_catalog_data()

    for g_issue in validate_immutable_source_refs(speech_data, cleanup_data, diarization_data):
        issues.append(ValidationIssue(g_issue.source, g_issue.message))

    for g_issue in validate_identities_and_integrity(speech_data, cleanup_data, diarization_data):
        issues.append(ValidationIssue(g_issue.source, g_issue.message))

    for g_issue in validate_release_safety(ROOT):
        issues.append(ValidationIssue(g_issue.source, g_issue.message))

    releases_data = load_all_releases_data()
    for g_issue in validate_no_executable_payloads(speech_data, cleanup_data, diarization_data, releases_data):
        issues.append(ValidationIssue(g_issue.source, g_issue.message))

    if CHANNELS_FILE.exists():
        channels_data = load_channels_data()
        for g_issue in validate_channel_and_release_pointers(channels_data, RELEASES_DIR, GENERATED_DIR):
            issues.append(ValidationIssue(g_issue.source, g_issue.message))

    for rel_id, rel_data in load_all_releases_data().items():
        for g_issue in validate_release_snapshot_integrity(rel_id, rel_data, GENERATED_DIR):
            issues.append(ValidationIssue(f"generated/releases/{rel_id}", g_issue.message))
        for g_issue in validate_redistribution_clearance(rel_data):
            issues.append(ValidationIssue(f"catalog/releases/{rel_id}.yaml", g_issue.message))

    # 3. Validate generated channel manifests with distribution schema
    dist_schema_path = SCHEMA_DIR / "channel_distribution.schema.json"
    if dist_schema_path.exists():
        dist_validator = Draft202012Validator(_load_json(dist_schema_path))
        for chan_file in (GENERATED_DIR / "channels").glob("*.json"):
            try:
                for error in dist_validator.iter_errors(_load_json(chan_file)):
                    location = ".".join(str(part) for part in error.absolute_path)
                    prefix = f"{location}: " if location else ""
                    issues.append(ValidationIssue(str(chan_file.relative_to(ROOT)), prefix + error.message))
            except (OSError, json.JSONDecodeError, TypeError, SchemaError) as exc:
                issues.append(ValidationIssue(str(chan_file.relative_to(ROOT)), str(exc)))

    # 4. Check generated file freshness
    for message in export_generated(check=True):
        issues.append(ValidationIssue("generated", message))

    # 5. Validate existing signatures against public key
    for g_issue in validate_signatures(GENERATED_DIR):
        issues.append(ValidationIssue(g_issue.source, g_issue.message))

    return issues
