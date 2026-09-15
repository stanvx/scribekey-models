from pathlib import Path

from scribekey_models.catalog import (
    GENERATED_DIR,
    export_generated,
    generate_all_artifacts,
    generate_bootstrap_catalog,
    validate,
)


def test_catalogue_validates() -> None:
    assert validate() == []


def test_generation_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    export_generated(first)
    export_generated(second)
    assert {
        path.name: path.read_bytes() for path in first.iterdir()
    } == {
        path.name: path.read_bytes() for path in second.iterdir()
    }


def test_expected_runtime_artifacts_are_generated() -> None:
    assert {artifact.filename for artifact in generate_all_artifacts()} == {
        "model_catalog.json",
        "speaker_diarization_manifest.json",
        "cleanup_model_catalog.json",
        "bootstrap_catalog.json",
    }


def test_bootstrap_identifies_public_source() -> None:
    assert generate_bootstrap_catalog()["generator"]["source"] == "scribekey-models"


def test_committed_generated_files_are_current() -> None:
    assert export_generated(GENERATED_DIR, check=True) == []

