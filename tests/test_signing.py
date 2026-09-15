from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from scribekey_models.signing import (
    generate_keypair,
    load_private_key,
    load_public_key,
    sign_file,
    verify_file,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_KEY_PEM = FIXTURES_DIR / "test_key.pem"
TEST_KEY_PUB = FIXTURES_DIR / "test_key.pub"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "signature.schema.json"


def test_ecdsa_p256_sign_and_verify_roundtrip(tmp_path: Path) -> None:
    priv, pub = generate_keypair()
    payload_file = tmp_path / "data.json"
    payload_file.write_text('{"status": "ok"}\n', encoding="utf-8")

    sig_file = sign_file(payload_file, priv, key_id="test-key-1")
    assert sig_file.exists()

    ok, msg = verify_file(payload_file, sig_file, public_key=pub)
    assert ok is True
    assert msg == "Signature is valid"


def test_signature_payload_matches_schema(tmp_path: Path) -> None:
    priv, _ = generate_keypair()
    payload_file = tmp_path / "channel.json"
    payload_file.write_text('{"channel": "qa"}\n', encoding="utf-8")

    sig_file = sign_file(payload_file, priv, key_id="scribekey-test")
    sig_data = json.loads(sig_file.read_text(encoding="utf-8"))

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(sig_data))
    assert errors == []
    assert sig_data["algorithm"] == "SHA256withECDSA"


def test_verify_detects_exact_byte_tampering(tmp_path: Path) -> None:
    priv, pub = generate_keypair()
    payload_file = tmp_path / "manifest.json"
    payload_file.write_text('{"version": "1.0.0"}\n', encoding="utf-8")

    sig_file = sign_file(payload_file, priv)

    # Tamper with file content by even a single byte (e.g. whitespace)
    payload_file.write_text('{"version": "1.0.0"} \n', encoding="utf-8")

    ok, msg = verify_file(payload_file, sig_file, public_key=pub)
    assert ok is False
    assert "Digest mismatch" in msg


def test_verify_detects_signature_tampering(tmp_path: Path) -> None:
    priv, pub = generate_keypair()
    payload_file = tmp_path / "manifest.json"
    payload_file.write_text('{"version": "1.0.0"}\n', encoding="utf-8")

    sig_file = sign_file(payload_file, priv)

    # Corrupt signature in sig_file while keeping valid base64
    sig_data = json.loads(sig_file.read_text(encoding="utf-8"))
    orig_bytes = bytearray(sig_data["signature"].encode("ascii"))
    orig_bytes[10] = ord("A") if orig_bytes[10] != ord("A") else ord("B")
    sig_data["signature"] = orig_bytes.decode("ascii")
    sig_file.write_text(json.dumps(sig_data), encoding="utf-8")

    ok, msg = verify_file(payload_file, sig_file, public_key=pub)
    assert ok is False
    assert "verification failed" in msg


def test_verify_fails_with_wrong_public_key(tmp_path: Path) -> None:
    priv1, _ = generate_keypair()
    _, pub2 = generate_keypair()
    payload_file = tmp_path / "file.json"
    payload_file.write_text("{}", encoding="utf-8")

    sig_file = sign_file(payload_file, priv1)
    ok, msg = verify_file(payload_file, sig_file, public_key=pub2)
    assert ok is False
    assert "verification failed" in msg


def test_verify_rejects_unsupported_algorithm(tmp_path: Path) -> None:
    priv, pub = generate_keypair()
    payload_file = tmp_path / "file.json"
    payload_file.write_text("{}", encoding="utf-8")

    sig_file = sign_file(payload_file, priv)
    sig_data = json.loads(sig_file.read_text(encoding="utf-8"))
    sig_data["algorithm"] = "rsa2048"
    sig_file.write_text(json.dumps(sig_data), encoding="utf-8")

    ok, msg = verify_file(payload_file, sig_file, public_key=pub)
    assert ok is False
    assert "Unsupported signature algorithm" in msg


def test_load_key_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    priv_pem = TEST_KEY_PEM.read_text(encoding="utf-8")
    monkeypatch.setenv("MODEL_RELEASE_SIGNING_KEY", priv_pem)

    loaded_priv = load_private_key(env_var="MODEL_RELEASE_SIGNING_KEY")
    loaded_pub = load_public_key(TEST_KEY_PUB)

    payload_file = tmp_path / "target.json"
    payload_file.write_text('{"test": true}', encoding="utf-8")
    sig_file = sign_file(payload_file, loaded_priv)

    ok, _ = verify_file(payload_file, sig_file, public_key=loaded_pub)
    assert ok is True
