from __future__ import annotations

from pathlib import Path

import pytest

from scribekey_models.guardrails import (
    validate_channel_and_release_pointers,
    validate_identities_and_integrity,
    validate_immutable_source_refs,
    validate_model_configuration,
    validate_no_executable_payloads,
    validate_redistribution_clearance,
    validate_release_safety,
)


def _speech_model(**overrides: object) -> dict:
    return {
        "id": "speech",
        "family": "PARAKEET",
        "sherpaConfig": {"type": "nemo_ctc"},
        "transcriptionMode": "SEGMENTED",
        "diskMb": 1,
        "files": [
            {"name": name, "sizeBytes": 1, "downloadUrl": "https://example.com/model"}
            for name in ("model.onnx", "tokens.txt")
        ],
        **overrides,
    }


def _cleanup_model(**overrides: object) -> dict:
    return {
        "modelId": "cleanup",
        "bundleFileName": "model.gguf",
        "revision": "a" * 40,
        "downloadUrl": f"https://huggingface.co/org/repo/resolve/{'a' * 40}/model.gguf",
        "maxOutputTokens": 256,
        "contextTokens": 4096,
        "deterministicDecoding": True,
        "topK": 1,
        "topP": 1.0,
        "temperature": 0.0,
        "repetitionPenalty": None,
        **overrides,
    }


def _diarization_data() -> dict:
    return {"models": [
        {"role": role, "fileName": f"{role}.onnx", "downloadUrl": "https://example.com/model"}
        for role in ("SEGMENTATION", "EMBEDDING")
    ]}


@pytest.mark.parametrize(("runtime", "family", "names"), [
    ("moonshine", "MOONSHINE", "preprocess.onnx encode.int8.onnx uncached_decode.int8.onnx cached_decode.int8.onnx tokens.txt"),
    ("moonshine_v2", "MOONSHINE", "encoder_model.ort decoder_model_merged.ort tokens.txt"),
    ("nemo_ctc", "PARAKEET", "model.onnx tokens.txt"),
    ("nemo_transducer", "PARAKEET", "encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt"),
    ("whisper", "DISTIL_WHISPER", "encoder.int8.onnx decoder.int8.onnx tokens.txt"),
    ("canary", "CANARY", "encoder.int8.onnx decoder.int8.onnx tokens.txt"),
    ("omnilingual_ctc", "OMNILINGUAL", "model.int8.onnx tokens.txt"),
    ("qwen3_asr", "QWEN3_ASR", "conv_frontend.onnx encoder.int8.onnx decoder.int8.onnx tokenizer/merges.txt tokenizer/tokenizer_config.json tokenizer/vocab.json"),
])
def test_runtime_layout_requires_android_consumer_files(runtime, family, names) -> None:
    model = _speech_model(
        family=family,
        sherpaConfig={"type": runtime},
        files=[{"name": name, "sizeBytes": 1, "downloadUrl": "https://example.com/model"}
               for name in names.split()],
    )
    assert validate_model_configuration({"models": [model]}, {}, _diarization_data()) == []
    removed = model["files"].pop()
    issues = validate_model_configuration({"models": [model]}, {}, _diarization_data())
    assert any("missing required files" in issue.message and removed["name"] in issue.message
               for issue in issues)


@pytest.mark.parametrize(("total_bytes", "disk_mb"), [(1048575, 1), (1048576, 1), (1048577, 2)])
def test_speech_disk_size_rounds_up_total_bytes_once(total_bytes, disk_mb) -> None:
    model = _speech_model(diskMb=disk_mb)
    model["files"][0]["sizeBytes"] = total_bytes - 1
    assert validate_model_configuration({"models": [model]}, {}, _diarization_data()) == []
    model["diskMb"] += 1
    issues = validate_model_configuration({"models": [model]}, {}, _diarization_data())
    assert any("diskMb" in issue.message for issue in issues)


@pytest.mark.parametrize("name", ["../model.onnx", "/model.onnx", "tokenizer/../model.onnx", "./model.onnx", "tokenizer//model.onnx", "tokenizer\\model.onnx", "model\u0000.onnx"])
def test_speech_install_path_must_be_safe_and_relative(name) -> None:
    model = _speech_model()
    model["files"][0]["name"] = name
    issues = validate_model_configuration({"models": [model]}, {}, _diarization_data())
    assert any("unsafe install path" in issue.message for issue in issues)


def test_speech_rejects_duplicate_files_family_mismatch_and_invalid_online_runtime() -> None:
    model = _speech_model(family="MOONSHINE", transcriptionMode="CACHE_AWARE_ONLINE")
    model["files"].append(model["files"][0].copy())
    issues = validate_model_configuration({"models": [model]}, {}, _diarization_data())
    assert any("Duplicate install path" in issue.message for issue in issues)
    assert any("family" in issue.message for issue in issues)
    assert any("CACHE_AWARE_ONLINE" in issue.message for issue in issues)


@pytest.mark.parametrize(("repository", "revision"), [("wrong/repo", "a" * 40), ("org/repo", "b" * 40)])
def test_speech_provenance_must_match_hugging_face_artifacts(repository, revision) -> None:
    model = _speech_model(provenance={"exportRepository": repository, "exportRevision": revision})
    for file in model["files"]:
        file["downloadUrl"] = f"https://huggingface.co/org/repo/resolve/{'a' * 40}/{file['name']}"
    issues = validate_model_configuration({"models": [model]}, {}, _diarization_data())
    assert any("does not match Hugging Face" in issue.message for issue in issues)
    model["provenance"] = {"exportRepository": "org/repo", "exportRevision": "a" * 40}
    assert validate_model_configuration({"models": [model]}, {}, _diarization_data()) == []


def test_cleanup_rejects_duplicate_ids_revision_mismatch_and_full_context_output() -> None:
    model = _cleanup_model(revision="b" * 40, maxOutputTokens=4096)
    issues = validate_model_configuration({}, {"production": model, "candidates": [model.copy()]}, _diarization_data())
    assert any("Duplicate cleanup model id" in issue.message for issue in issues)
    assert any("does not match Hugging Face" in issue.message for issue in issues)
    assert any("contextTokens must exceed maxOutputTokens" in issue.message for issue in issues)


@pytest.mark.parametrize("overrides", [{"topK": 2}, {"topP": 0.9}, {"temperature": 0.1}, {"repetitionPenalty": 1.0}])
def test_deterministic_cleanup_requires_greedy_sampling(overrides) -> None:
    model = _cleanup_model(**overrides)
    issues = validate_model_configuration({}, {"production": model}, _diarization_data())
    assert any("deterministicDecoding" in issue.message for issue in issues)
    model["deterministicDecoding"] = False
    assert validate_model_configuration({}, {"production": model}, _diarization_data()) == []


def test_cleanup_and_diarization_install_names_cannot_contain_paths() -> None:
    diarization = _diarization_data()
    diarization["models"][0]["fileName"] = "../model.onnx"
    issues = validate_model_configuration({}, {"production": _cleanup_model(bundleFileName="nested/model.gguf")}, diarization)
    assert sum("unsafe install path" in issue.message for issue in issues) == 2


def test_diarization_requires_one_model_for_each_role() -> None:
    diarization = _diarization_data()
    diarization["models"][1]["role"] = "SEGMENTATION"
    issues = validate_model_configuration({}, {}, diarization)
    assert any("exactly one SEGMENTATION" in issue.message for issue in issues)
    assert any("exactly one EMBEDDING" in issue.message for issue in issues)


def test_diarization_hugging_face_source_identity_must_match() -> None:
    diarization = _diarization_data()
    diarization["models"][0].update(
        downloadUrl=f"https://huggingface.co/org/repo/resolve/{'a' * 40}/model.onnx",
        sourceRepository="org/repo", sourceRevision="b" * 40,
    )
    issues = validate_model_configuration({}, {}, diarization)
    assert any("does not match Hugging Face" in issue.message for issue in issues)


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


def test_guardrail_checks_every_fallback_source_for_immutability() -> None:
    speech_data = {
        "models": [
            {
                "id": "fallback-model",
                "files": [
                    {
                        "name": "model.onnx",
                        "downloadUrl": "https://example.com/official/model.onnx",
                        "downloadUrls": [
                            "https://huggingface.co/org/repo/resolve/main/model.onnx"
                        ],
                        "sha256": "a" * 64,
                        "sizeBytes": 100,
                    }
                ],
            }
        ]
    }

    issues = validate_immutable_source_refs(speech_data, {}, {})

    assert any("uses mutable ref 'main'" in issue.message for issue in issues)


def test_guardrail_requires_https_for_fallback_sources() -> None:
    speech_data = {
        "models": [
            {
                "id": "fallback-model",
                "files": [
                    {
                        "name": "model.onnx",
                        "downloadUrl": "https://example.com/official/model.onnx",
                        "downloadUrls": ["http://mirror.example/model.onnx"],
                        "sha256": "a" * 64,
                        "sizeBytes": 100,
                    }
                ],
            }
        ]
    }

    issues = validate_immutable_source_refs(speech_data, {}, {})

    assert any("download source must use HTTPS" in issue.message for issue in issues)


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
