from __future__ import annotations

from pathlib import Path

from scribekey_models.guardrails import (
    validate_channel_and_release_pointers,
    validate_identities_and_integrity,
    validate_immutable_source_refs,
    validate_no_executable_payloads,
    validate_redistribution_clearance,
    validate_release_safety,
)


def test_guardrail_rejects_mutable_ref_in_speech_download_url() -> None:
    speech_data = {
        "models": [
            {
                "id": "bad-model",
                "files": [
                    {
                        "name": "model.onnx",
                        "downloadUrl": "https://huggingface.co/org/repo/resolve/main/model.onnx",
                        "sha256": "a" * 64,
                        "sizeBytes": 100,
                    }
                ],
            }
        ]
    }
    issues = validate_immutable_source_refs(speech_data, {}, {})
    assert any("uses mutable ref 'main'" in issue.message for issue in issues)


def test_guardrail_rejects_non_hex_hf_commit_in_download_url() -> None:
    speech_data = {
        "models": [
            {
                "id": "bad-model",
                "files": [
                    {
                        "name": "model.onnx",
                        "downloadUrl": "https://huggingface.co/org/repo/resolve/v1.0-tag/model.onnx",
                        "sha256": "a" * 64,
                        "sizeBytes": 100,
                    }
                ],
            }
        ]
    }
    issues = validate_immutable_source_refs(speech_data, {}, {})
    assert any("must use a 40-character commit hash" in issue.message for issue in issues)


def test_guardrail_rejects_mutable_revision_in_cleanup_and_diarization() -> None:
    cleanup_data = {
        "production": {
            "modelId": "cleanup-prod",
            "revision": "main",
            "downloadUrl": "https://example.com/model.gguf",
            "sha256": "a" * 64,
            "sizeBytes": 100,
        }
    }
    diarization_data = {
        "models": [
            {
                "role": "SEGMENTATION",
                "sourceRevision": "master",
                "downloadUrl": "https://example.com/model.onnx",
                "sha256": "a" * 64,
                "sizeBytes": 100,
            }
        ]
    }
    issues = validate_immutable_source_refs({}, cleanup_data, diarization_data)
    assert any("cannot be mutable ref 'main'" in issue.message for issue in issues)
    assert any("cannot be mutable ref 'master'" in issue.message for issue in issues)


def test_guardrail_rejects_invalid_sha256_or_negative_size() -> None:
    speech_data = {
        "models": [
            {
                "id": "bad-sha",
                "files": [
                    {
                        "name": "model.onnx",
                        "downloadUrl": "https://example.com/model.onnx",
                        "sha256": "NOT-A-HEX-STRING",
                        "sizeBytes": -50,
                    }
                ],
            }
        ]
    }
    issues = validate_identities_and_integrity(speech_data, {}, {})
    assert any("invalid sha256" in issue.message for issue in issues)
    assert any("invalid sizeBytes" in issue.message for issue in issues)


def test_guardrail_rejects_duplicate_model_ids() -> None:
    speech_data = {
        "models": [
            {
                "id": "model-1",
                "files": [{"name": "a", "downloadUrl": "url", "sha256": "a" * 64, "sizeBytes": 10}],
            },
            {
                "id": "model-1",
                "files": [{"name": "b", "downloadUrl": "url", "sha256": "b" * 64, "sizeBytes": 10}],
            },
        ]
    }
    issues = validate_identities_and_integrity(speech_data, {}, {})
    assert any("Duplicate speech model id: 'model-1'" in issue.message for issue in issues)


def test_guardrail_rejects_invalid_replacement_id() -> None:
    speech_data = {
        "models": [
            {
                "id": "legacy-model",
                "replacementId": "does-not-exist",
                "files": [{"name": "a", "downloadUrl": "url", "sha256": "a" * 64, "sizeBytes": 10}],
            }
        ]
    }
    issues = validate_identities_and_integrity(speech_data, {}, {})
    assert any("replacementId 'does-not-exist' not found" in issue.message for issue in issues)


def test_guardrail_rejects_model_binary_in_repo(tmp_path: Path) -> None:
    bad_binary = tmp_path / "model.onnx"
    bad_binary.write_bytes(b"\x08\x01\x12\x04test")

    issues = validate_release_safety(tmp_path)
    assert any("Forbidden model binary file" in issue.message for issue in issues)


def test_guardrail_rejects_un_cleared_recovery_assets() -> None:
    release_data = {
        "recoveryAssets": {
            "clearedModels": [
                {
                    "modelId": "proprietary-model",
                    "cleared": False,
                    "licenseId": "Proprietary",
                }
            ]
        }
    }
    issues = validate_redistribution_clearance(release_data)
    assert any("cleared is false" in issue.message for issue in issues)
    assert any("is not in redistribution-cleared permissive licenses" in issue.message for issue in issues)


def test_guardrail_rejects_executable_payload_extensions() -> None:
    speech_data = {
        "models": [
            {
                "id": "exploit-model",
                "files": [
                    {
                        "name": "malicious.apk",
                        "downloadUrl": "https://example.com/malicious.apk",
                        "sha256": "a" * 64,
                        "sizeBytes": 100,
                    }
                ],
            }
        ]
    }
    issues = validate_no_executable_payloads(speech_data, {}, {})
    assert any("executable payload 'malicious.apk' (.apk)" in issue.message for issue in issues)


def test_guardrail_rejects_channel_pointing_to_missing_release(tmp_path: Path) -> None:
    base_comp = {
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
    channels_data = {
        "channels": {
            "qa": {
                "targetRelease": "non-existent-release",
                "sequence": 1,
                "issuedAt": "2026-09-15T11:00:00Z",
                "compatibility": dict(base_comp),
            },
            "stable": {
                "targetRelease": "also-missing",
                "sequence": 1,
                "issuedAt": "2026-09-15T11:00:00Z",
                "compatibility": dict(base_comp),
            },
        }
    }
    releases_dir = tmp_path / "releases"
    releases_dir.mkdir()
    issues = validate_channel_and_release_pointers(channels_data, releases_dir, tmp_path)
    assert any("points to non-existent release" in issue.message for issue in issues)


def test_guardrail_rejects_invalid_sequence_and_unknown_runtime_family(tmp_path: Path) -> None:
    rel_file = tmp_path / "releases" / "2026.09.1.yaml"
    rel_file.parent.mkdir(parents=True, exist_ok=True)
    rel_file.write_text("{}", encoding="utf-8")

    channels_data = {
        "channels": {
            "qa": {
                "targetRelease": "2026.09.1",
                "sequence": 0,  # Invalid sequence (< 1)
                "issuedAt": "2026-09-15T11:00:00Z",
                "compatibility": {
                    "minAndroidApiLevel": 28,
                    "minAppVersionCode": 10,
                    "catalogsSchemaVersion": 1,
                    "supportedRuntimeFamilies": ["unknown_runtime"],
                    "supportedConfigFamilies": ["speech-model-catalog"],
                },
            },
            "stable": {
                "targetRelease": "2026.09.1",
                "sequence": 1,
                "issuedAt": "2026-09-15T11:00:00Z",
                "compatibility": {
                    "minAndroidApiLevel": 16,  # Invalid API level (< 21)
                    "minAppVersionCode": 10,
                    "catalogsSchemaVersion": 1,
                    "supportedRuntimeFamilies": ["sherpa-onnx"],
                    "supportedConfigFamilies": ["speech-model-catalog"],
                },
            },
        }
    }
    issues = validate_channel_and_release_pointers(channels_data, tmp_path / "releases", tmp_path)
    assert any("invalid anti-rollback sequence '0'" in issue.message for issue in issues)
    assert any("unknown runtime families" in issue.message for issue in issues)
    assert any("minAndroidApiLevel must be integer >= 21" in issue.message for issue in issues)
