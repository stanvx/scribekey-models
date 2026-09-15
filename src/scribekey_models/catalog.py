from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

ROOT = Path(__file__).resolve().parents[2]
CATALOG_DIR = ROOT / "catalog"
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
    return [
        GenerationArtifact("model_catalog.json", _render_json(generate_speech_catalog())),
        GenerationArtifact(
            "speaker_diarization_manifest.json",
            _render_json(generate_diarization_manifest()),
        ),
        GenerationArtifact("cleanup_model_catalog.json", _render_json(generate_cleanup_catalog())),
    ]


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
        output_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(artifact.content, encoding="utf-8")
    return drift


def validate() -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    canonical = [
        (CATALOG_DIR / "speech.yaml", SCHEMA_DIR / "speech.schema.json"),
        (CATALOG_DIR / "cleanup.yaml", SCHEMA_DIR / "cleanup.schema.json"),
        (CATALOG_DIR / "diarization.yaml", SCHEMA_DIR / "diarization.schema.json"),
    ]
    for source, schema in canonical:
        try:
            validator = Draft202012Validator(_load_json(schema))
            for error in validator.iter_errors(_load_yaml(source)):
                location = ".".join(str(part) for part in error.absolute_path)
                prefix = f"{location}: " if location else ""
                issues.append(ValidationIssue(str(source), prefix + error.message))
        except (OSError, TypeError, ValueError, yaml.YAMLError, SchemaError) as exc:
            issues.append(ValidationIssue(str(source), str(exc)))

    for message in export_generated(check=True):
        issues.append(ValidationIssue("generated", message))
    return issues

