import datetime
import json
from pathlib import Path

import scribekey_models.health as health_module
from scribekey_models.health import (
    SourceMetadata,
    _sha256_from_headers,
    check_channel_sources,
    load_channel_targets,
    probe_source_metadata,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def test_stable_targets_include_primary_and_fallback_sources_in_order(tmp_path: Path) -> None:
    _write_distribution(tmp_path)

    release_id, targets = load_channel_targets(tmp_path)

    assert release_id == "2026.09.1"
    speech_sources = [target for target in targets if target.model_id == "speech-model"]
    assert [target.url for target in speech_sources] == [
        "https://official.example/model.onnx",
        "https://mirror.example/model.onnx",
        "https://recovery.example/model.onnx",
    ]
    assert [target.source_index for target in speech_sources] == [0, 1, 2]
    assert all(target.expected_size_bytes == 4 for target in speech_sources)
    assert all(target.expected_sha256 == SHA_A for target in speech_sources)
    assert {target.catalog for target in targets} == {"speech", "cleanup", "diarization"}



def test_health_target_loading_fails_closed_for_empty_published_catalog(tmp_path: Path) -> None:
    _write_distribution(tmp_path)
    release_dir = tmp_path / "generated" / "releases" / "2026.09.1"
    _write_json(release_dir / "model_catalog.json", {"models": []})

    import pytest

    with pytest.raises(ValueError, match="speech catalog contains no model artifact sources"):
        load_channel_targets(tmp_path)

def test_health_report_flags_size_and_checksum_mismatches(tmp_path: Path) -> None:
    _write_distribution(tmp_path)

    def probe(url: str, _timeout: float) -> SourceMetadata:
        if "mirror.example" in url:
            return SourceMetadata(200, 5, SHA_A)
        if "recovery.example" in url:
            return SourceMetadata(200, 4, SHA_B)
        return SourceMetadata(200, 4 if "model.onnx" in url else 8, SHA_A)

    report = check_channel_sources(
        tmp_path,
        probe=probe,
        now=datetime.datetime(2026, 9, 16, tzinfo=datetime.UTC),
    )

    assert not report.healthy
    assert len(report.failures) == 2
    assert "size mismatch" in report.failures[0].reason
    assert "SHA-256 mismatch" in report.failures[1].reason
    assert not report.failures[1].checksum_verified
    assert report.incident_marker == "<!-- scribekey-model-source-health:stable:2026.09.1 -->"
    assert report.incident_marker in report.incident_body()
    assert "Distribution channel: `stable`" in report.incident_body()




def test_health_report_rejects_empty_success_response(tmp_path: Path) -> None:
    _write_distribution(tmp_path)

    report = check_channel_sources(
        tmp_path,
        probe=lambda _url, _timeout: SourceMetadata(204, None, None),
    )

    assert not report.healthy
    assert all("unexpected HTTP status 204" in failure.reason for failure in report.failures)

def test_health_report_records_malformed_source_as_failure(tmp_path: Path) -> None:
    _write_distribution(tmp_path)

    def malformed_probe(_url: str, _timeout: float) -> SourceMetadata:
        raise ValueError("invalid URL")

    report = check_channel_sources(tmp_path, probe=malformed_probe)

    assert not report.healthy
    assert all("source unavailable: invalid URL" in failure.reason for failure in report.failures)

def test_incident_identity_is_stable_when_failure_details_change(tmp_path: Path) -> None:
    _write_distribution(tmp_path)

    first = check_channel_sources(
        tmp_path,
        probe=lambda _url, _timeout: SourceMetadata(503, None, None),
        now=datetime.datetime(2026, 9, 16, tzinfo=datetime.UTC),
    )
    second = check_channel_sources(
        tmp_path,
        probe=lambda _url, _timeout: SourceMetadata(200, 99, None),
        now=datetime.datetime(2026, 9, 17, tzinfo=datetime.UTC),
    )

    assert first.incident_key == second.incident_key
    assert first.incident_title == second.incident_title


def test_provider_checksum_headers_support_hugging_face_and_digest() -> None:
    assert _sha256_from_headers({"X-Linked-Etag": f'"{SHA_A}"'}) == SHA_A
    assert (
        _sha256_from_headers(
            {"Digest": "sha-256=qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqo="}
        )
        == SHA_A
    )
    assert _sha256_from_headers({"ETag": SHA_B}) is None


def test_probe_falls_back_to_range_get_when_head_is_unsupported(monkeypatch) -> None:
    range_calls: list[str] = []
    monkeypatch.setattr(
        health_module,
        "_request_without_redirects",
        lambda _url, _timeout: (405, {"Content-Length": "123"}),
    )

    def range_probe(url: str, _timeout: float):
        range_calls.append(url)
        return 206, {"Content-Range": "bytes 0-0/4", "X-Checksum-Sha256": SHA_A}

    monkeypatch.setattr(health_module, "_range_probe", range_probe)

    metadata = probe_source_metadata("https://mirror.example/model.onnx", 5.0)

    assert range_calls == ["https://mirror.example/model.onnx"]
    assert metadata == SourceMetadata(206, 4, SHA_A)


def _write_distribution(root: Path) -> None:
    channel_dir = root / "generated" / "channels"
    release_dir = root / "generated" / "releases" / "2026.09.1"
    channel_dir.mkdir(parents=True)
    release_dir.mkdir(parents=True)
    _write_json(
        channel_dir / "stable.json",
        {
            "releaseId": "2026.09.1",
            "catalogs": {
                "speech": {"path": "../releases/2026.09.1/model_catalog.json"},
                "cleanup": {"path": "../releases/2026.09.1/cleanup_model_catalog.json"},
                "diarization": {
                    "path": "../releases/2026.09.1/speaker_diarization_manifest.json"
                },
            },
        },
    )
    _write_json(
        release_dir / "model_catalog.json",
        {
            "models": [
                {
                    "id": "speech-model",
                    "files": [
                        {
                            "name": "model.onnx",
                            "downloadUrl": "https://official.example/model.onnx",
                            "downloadUrls": [
                                "https://mirror.example/model.onnx",
                                "https://recovery.example/model.onnx",
                            ],
                            "sizeBytes": 4,
                            "sha256": SHA_A,
                        }
                    ],
                }
            ]
        },
    )
    _write_json(
        release_dir / "cleanup_model_catalog.json",
        {
            "production": {
                "modelId": "cleanup-model",
                "bundleFileName": "cleanup.gguf",
                "downloadUrl": "https://official.example/cleanup.gguf",
                "sizeBytes": 8,
                "sha256": SHA_A,
            }
        },
    )
    _write_json(
        release_dir / "speaker_diarization_manifest.json",
        {
            "models": [
                {
                    "role": "SEGMENTATION",
                    "fileName": "segmentation.onnx",
                    "downloadUrl": "https://official.example/segmentation.onnx",
                    "sizeBytes": 8,
                    "sha256": SHA_A,
                }
            ]
        },
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
