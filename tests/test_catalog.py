from pathlib import Path

import scribekey_models.catalog as catalog_module
from scribekey_models.catalog import (
    GENERATED_DIR,
    export_generated,
    generate_all_artifacts,
    generate_cleanup_catalog,
    generate_diarization_manifest,
    generate_speech_catalog,
    validate,
)
from scribekey_models.cli import _parser


def test_catalogue_validates() -> None:
    assert validate() == []


def test_generation_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    export_generated(first)
    export_generated(second)
    assert {
        str(path.relative_to(first)): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    } == {
        str(path.relative_to(second)): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }


def test_expected_runtime_artifacts_are_generated() -> None:
    filenames = {artifact.filename for artifact in generate_all_artifacts()}
    assert {
        "model_catalog.json",
        "speaker_diarization_manifest.json",
        "cleanup_model_catalog.json",
    }.issubset(filenames)
    assert "channels/qa.json" in filenames
    assert "channels/stable.json" in filenames
    assert "releases/2026.09.1/release.json" in filenames


def test_committed_generated_files_are_current() -> None:
    assert export_generated(GENERATED_DIR, check=True) == []


def test_generate_cli_accepts_consumer_output_directory(tmp_path: Path) -> None:
    args = _parser().parse_args(["generate", "--output-dir", str(tmp_path), "--check"])
    assert args.output_dir == tmp_path
    assert args.check is True


def test_speech_refresh_keeps_legacy_ids_and_adds_new_runtime_shapes() -> None:
    models = {model["id"]: model for model in generate_speech_catalog()["models"]}

    assert models["moonshine-tiny"]["deprecated"] is True
    assert models["moonshine-tiny"]["replacementId"] == "moonshine-v2-tiny-en"
    assert models["moonshine-base"]["deprecated"] is True
    assert models["moonshine-base"]["replacementId"] == "moonshine-v2-base-en"

    assert models["moonshine-v2-tiny-en"]["sherpaConfig"]["type"] == "moonshine_v2"
    assert models["moonshine-v2-base-en"]["sherpaConfig"]["type"] == "moonshine_v2"
    assert models["omnilingual-asr-300m"]["sherpaConfig"]["type"] == "omnilingual_ctc"
    assert models["qwen3-asr-0.6b"]["sherpaConfig"]["type"] == "qwen3_asr"

    assert all(
        "/resolve/main/" not in file["downloadUrl"]
        for model in models.values()
        for file in model["files"]
    )


def test_runtime_catalogues_preserve_ordered_fallback_sources(monkeypatch) -> None:
    speech = catalog_module.load_speech_catalog_data()
    speech["models"][0]["files"][0]["downloadUrls"] = [
        "https://mirror.example/model.onnx",
        "https://github.example/model.onnx",
    ]
    monkeypatch.setattr(catalog_module, "load_speech_catalog_data", lambda: speech)
    assert generate_speech_catalog()["models"][0]["files"][0]["downloadUrls"] == [
        "https://mirror.example/model.onnx",
        "https://github.example/model.onnx",
    ]

    cleanup = catalog_module.load_cleanup_catalog_data()
    cleanup["production"]["downloadUrls"] = ["https://mirror.example/cleanup.gguf"]
    monkeypatch.setattr(catalog_module, "load_cleanup_catalog_data", lambda: cleanup)
    assert generate_cleanup_catalog()["production"]["downloadUrls"] == [
        "https://mirror.example/cleanup.gguf"
    ]

    diarization = catalog_module.load_diarization_catalog_data()
    diarization["models"][0]["downloadUrls"] = ["https://mirror.example/diarization.onnx"]
    monkeypatch.setattr(catalog_module, "load_diarization_catalog_data", lambda: diarization)
    assert generate_diarization_manifest()["models"][0]["downloadUrls"] == [
        "https://mirror.example/diarization.onnx"
    ]

