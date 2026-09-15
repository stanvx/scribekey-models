from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scribekey_models.mirror import (
    check_mirror_configuration,
    plan_release_mirror,
)
from scribekey_models.promotion import (
    promote,
)
from scribekey_models.signing import (
    generate_keypair,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_worktree(tmp_path: Path) -> Path:
    """Create a minimal mock repo worktree with canonical catalog, schemas, keys, and generated dirs."""
    root = tmp_path / "repo"
    root.mkdir()

    # Source directories
    catalog_dir = root / "catalog"
    releases_dir = catalog_dir / "releases"
    releases_dir.mkdir(parents=True)
    schemas_dir = root / "schemas"
    schemas_dir.mkdir()
    generated_dir = root / "generated" / "channels"
    generated_dir.mkdir(parents=True)
    keys_dir = root / "keys"
    keys_dir.mkdir()

    # Copy actual repo schemas
    actual_root = Path(__file__).resolve().parents[1]
    for s in actual_root.glob("schemas/*.json"):
        shutil.copyfile(s, schemas_dir / s.name)

    # Generate test signing keys
    priv, pub = generate_keypair()
    from scribekey_models.signing import export_public_key_pem
    (keys_dir / "release-signing.pub").write_text(export_public_key_pem(pub), encoding="utf-8")

    comp_fixture = {
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

    # Release 1 definition
    rel_1 = {
        "schema_version": 1,
        "releaseId": "2026.09.1",
        "createdAt": "2026-09-15T10:00:00Z",
        "description": "First qualified release",
        "catalogs": {
            "speech": {"filename": "model_catalog.json", "sha256": "1" * 64, "sizeBytes": 1000},
            "diarization": {"filename": "speaker_diarization_manifest.json", "sha256": "2" * 64, "sizeBytes": 200},
            "cleanup": {"filename": "cleanup_model_catalog.json", "sha256": "3" * 64, "sizeBytes": 300},
        },
    }
    (releases_dir / "2026.09.1.yaml").write_text(yaml.safe_dump(rel_1), encoding="utf-8")

    # Release 2 definition
    rel_2 = {
        "schema_version": 1,
        "releaseId": "2026.09.2",
        "createdAt": "2026-09-16T10:00:00Z",
        "description": "Second qualified release",
        "catalogs": {
            "speech": {"filename": "model_catalog.json", "sha256": "5" * 64, "sizeBytes": 1100},
            "diarization": {"filename": "speaker_diarization_manifest.json", "sha256": "6" * 64, "sizeBytes": 210},
            "cleanup": {"filename": "cleanup_model_catalog.json", "sha256": "7" * 64, "sizeBytes": 310},
        },
    }
    (releases_dir / "2026.09.2.yaml").write_text(yaml.safe_dump(rel_2), encoding="utf-8")

    # Initial channels.yaml
    channels = {
        "schema_version": 1,
        "channels": {
            "qa": {
                "targetRelease": "2026.09.1",
                "sequence": 1,
                "issuedAt": "2026-09-15T10:00:00Z",
                "compatibility": comp_fixture,
            },
            "stable": {
                "targetRelease": "2026.09.1",
                "sequence": 1,
                "issuedAt": "2026-09-15T10:00:00Z",
                "compatibility": comp_fixture,
            },
        },
    }
    (catalog_dir / "channels.yaml").write_text(yaml.safe_dump(channels), encoding="utf-8")

    # Initial promotion to generate channel files and signatures
    promote("stable", "2026.09.1", sequence=1, compatibility=comp_fixture, private_key=priv, root=root)
    promote("qa", "2026.09.1", sequence=1, compatibility=comp_fixture, private_key=priv, root=root)

    return root


def test_promotion_is_idempotent(repo_worktree: Path) -> None:
    # 1. Re-running promote without key on existing release is idempotent no-op
    res1 = promote("stable", "2026.09.1", root=repo_worktree)
    assert res1.updated is False
    assert "already pointing at release '2026.09.1'" in res1.message
    assert res1.sequence == 1

    # 2. Promoting with same key is also an idempotent no-op (no re-signing or timestamp regeneration)
    stable_json_before = (repo_worktree / "generated" / "channels" / "stable.json").read_bytes()
    res2 = promote("stable", "2026.09.1", root=repo_worktree)
    assert res2.updated is False
    assert (repo_worktree / "generated" / "channels" / "stable.json").read_bytes() == stable_json_before


def test_promotion_enforces_monotonic_sequence(repo_worktree: Path) -> None:
    priv, _ = generate_keypair()

    # Attempting to promote with sequence < current sequence (1) must fail
    with pytest.raises(ValueError, match="strictly monotonically increasing"):
        promote("stable", "2026.09.2", sequence=0, private_key=priv, root=repo_worktree)

    # Automatic sequence promotion increments to 2
    res = promote("stable", "2026.09.2", private_key=priv, root=repo_worktree)
    assert res.updated is True
    assert res.sequence == 2

    manifest = json.loads((repo_worktree / "generated" / "channels" / "stable.json").read_text(encoding="utf-8"))
    assert manifest["sequence"] == 2
    assert manifest["releaseId"] == "2026.09.2"


def test_promotion_equal_sequence_with_changes_is_rejected(repo_worktree: Path) -> None:
    priv, _ = generate_keypair()

    # 1. Changing target release at same sequence must be rejected
    with pytest.raises(ValueError, match="Cannot modify target release"):
        promote("stable", "2026.09.2", sequence=1, private_key=priv, root=repo_worktree)

    # 2. Changing notes at same sequence must be rejected
    with pytest.raises(ValueError, match="Cannot modify notes"):
        promote("stable", "2026.09.1", sequence=1, notes="Modified notes at same sequence", private_key=priv, root=repo_worktree)

    # 3. Changing compatibility at same sequence must be rejected
    modified_comp = {
        "minAndroidApiLevel": 30,
        "minAppVersionCode": 10,
        "catalogsSchemaVersion": 1,
        "supportedRuntimeFamilies": ["sherpa-onnx", "gguf", "pyannote"],
        "supportedConfigFamilies": ["speech-model-catalog", "cleanup-model-catalog", "speaker-diarization-manifest"],
    }
    with pytest.raises(ValueError, match="Cannot modify compatibility"):
        promote("stable", "2026.09.1", sequence=1, compatibility=modified_comp, private_key=priv, root=repo_worktree)


def test_promotion_emergency_rollback_uses_higher_sequence(repo_worktree: Path) -> None:
    priv, _ = generate_keypair()

    # 1. Promote stable to 2026.09.2 (sequence 2)
    res2 = promote("stable", "2026.09.2", private_key=priv, root=repo_worktree)
    assert res2.sequence == 2

    # 2. Emergency rollback to 2026.09.1: must use a NEW HIGHER sequence (sequence 3)
    res_rollback = promote("stable", "2026.09.1", notes="Emergency rollback", private_key=priv, root=repo_worktree)
    assert res_rollback.updated is True
    assert res_rollback.sequence == 3

    manifest = json.loads((repo_worktree / "generated" / "channels" / "stable.json").read_text(encoding="utf-8"))
    assert manifest["sequence"] == 3
    assert manifest["releaseId"] == "2026.09.1"
    assert manifest["catalogs"]["speech"]["sha256"] == "1" * 64  # points to known-good 2026.09.1 artifact


def test_failed_publish_preserves_previous_stable(repo_worktree: Path) -> None:
    stable_file = repo_worktree / "generated" / "channels" / "stable.json"
    sig_file = repo_worktree / "generated" / "channels" / "stable.json.sig"

    original_manifest = stable_file.read_text(encoding="utf-8")
    original_sig = sig_file.read_text(encoding="utf-8")

    # Attempt promotion with non-existent release definition -> fails before staging commit
    with pytest.raises(FileNotFoundError):
        promote("stable", "non-existent-999", root=repo_worktree)

    # Verify previous stable remains completely untouched
    assert stable_file.read_text(encoding="utf-8") == original_manifest
    assert sig_file.read_text(encoding="utf-8") == original_sig

    # Attempt promotion with an invalid release definition (schema error)
    bad_rel_file = repo_worktree / "catalog" / "releases" / "2026.99.1.yaml"
    bad_rel_file.write_text("invalid: [not a valid release manifest]", encoding="utf-8")
    with pytest.raises(ValueError):
        promote("stable", "2026.99.1", root=repo_worktree)

    # Stable is still preserved
    assert stable_file.read_text(encoding="utf-8") == original_manifest
    assert sig_file.read_text(encoding="utf-8") == original_sig


def test_qa_and_stable_pointer_semantics(repo_worktree: Path) -> None:
    priv, _ = generate_keypair()

    # Promote QA to 2026.09.2 while keeping stable on 2026.09.1
    promote("qa", "2026.09.2", private_key=priv, root=repo_worktree)

    qa_manifest = json.loads((repo_worktree / "generated" / "channels" / "qa.json").read_text(encoding="utf-8"))
    stable_manifest = json.loads((repo_worktree / "generated" / "channels" / "stable.json").read_text(encoding="utf-8"))

    assert qa_manifest["releaseId"] == "2026.09.2"
    assert stable_manifest["releaseId"] == "2026.09.1"
    # Neither mutates artifact definitions, each references immutable relative release catalog paths
    assert qa_manifest["catalogs"]["speech"]["path"] == "../releases/2026.09.2/model_catalog.json"
    assert stable_manifest["catalogs"]["speech"]["path"] == "../releases/2026.09.1/model_catalog.json"


def test_channel_catalog_paths_are_relative_to_channel_document() -> None:
    """Channel catalog paths MUST be ../releases/<rel_id>/<filename> so relative URI resolution succeeds."""
    stable_path = ROOT / "generated" / "channels" / "stable.json"
    qa_path = ROOT / "generated" / "channels" / "qa.json"

    for chan_path in [stable_path, qa_path]:
        data = json.loads(chan_path.read_text(encoding="utf-8"))
        release_id = data["releaseId"]
        for cat_info in data["catalogs"].values():
            rel_path = cat_info["path"]
            assert rel_path.startswith(f"../releases/{release_id}/"), (
                f"Unexpected path format in {chan_path.name}: {rel_path}"
            )
            # Resolve relative to channel directory
            target_file = (chan_path.parent / rel_path).resolve()
            assert target_file.is_file(), f"Target catalog not found at {target_file} from {chan_path.name}"


def test_channel_compatibility_contract() -> None:
    """Verify channel distribution manifests satisfy the exact Android consumer contract."""
    stable_path = ROOT / "generated" / "channels" / "stable.json"
    data = json.loads(stable_path.read_text(encoding="utf-8"))

    comp = data["compatibility"]
    assert comp["minAndroidApiLevel"] == 28
    assert comp["minAppVersionCode"] == 10
    assert comp["catalogsSchemaVersion"] == 1
    assert comp["supportedRuntimeFamilies"] == ["sherpa-onnx", "gguf", "pyannote"]
    assert comp["supportedConfigFamilies"] == [
        "speech-model-catalog",
        "cleanup-model-catalog",
        "speaker-diarization-manifest",
    ]


def test_production_releases_have_no_unevidenced_recovery_claims() -> None:
    """Ensure production release and channel outputs assert no unevidenced recoveryAssets claims."""
    rel_data = yaml.safe_load((ROOT / "catalog" / "releases" / "2026.09.1.yaml").read_text(encoding="utf-8"))
    assert "recoveryAssets" not in rel_data

    stable_data = json.loads((ROOT / "generated" / "channels" / "stable.json").read_text(encoding="utf-8"))
    assert "recoveryAssets" not in stable_data

    qa_data = json.loads((ROOT / "generated" / "channels" / "qa.json").read_text(encoding="utf-8"))
    assert "recoveryAssets" not in qa_data


def test_cli_promotion_fails_closed_without_signing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The promote CLI command must fail closed when no signing key is provided."""
    monkeypatch.delenv("MODEL_RELEASE_SIGNING_KEY", raising=False)
    cmd = [
        sys.executable,
        "-c",
        "from scribekey_models.cli import main; main()",
        "promote",
        "--channel",
        "stable",
        "--release",
        "2026.09.1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    assert "Promotion must be signed" in proc.stdout or "Promotion must be signed" in proc.stderr


def test_promotion_update_requires_signing_key(repo_worktree: Path) -> None:
    with pytest.raises(ValueError, match="require a signing key"):
        promote("stable", "2026.09.2", root=repo_worktree)


def test_mirror_plan_generation_bounded() -> None:
    mock_release = {
        "releaseId": "test-rel",
        "recoveryAssets": {
            "clearedModels": [
                {
                    "modelId": "moonshine-v2",
                    "cleared": True,
                    "licenseId": "MIT",
                    "files": [
                        {
                            "name": "model.onnx",
                            "sha256": "a" * 64,
                            "upstreamUrl": "https://huggingface.co/csukuangfj2/sherpa/resolve/d1e6c30921780b8508d04b492dfb3ce8a51605d4/model.onnx",
                        }
                    ],
                }
            ]
        },
    }
    plans = plan_release_mirror(mock_release)
    assert len(plans) == 1
    assert plans[0].model_id == "moonshine-v2"
    assert plans[0].license_id == "MIT"
    assert "d1e6c30921780b8508d04b492dfb3ce8a51605d4" in plans[0].upstream_url

    status = check_mirror_configuration()
    assert status["mode"] == "dry_run"
    assert "No large model weights uploaded" in status["message"]
