from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KEY_ID = "scribekey-release-2026"
DEFAULT_PUBLIC_KEY_PATH = ROOT / "keys" / "release-signing.pub"


def generate_keypair() -> tuple[ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey]:
    private_key = ed25519.Ed25519PrivateKey.generate()
    return private_key, private_key.public_key()


def export_public_key_pem(public_key: ed25519.Ed25519PublicKey) -> str:
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")


def export_private_key_pem(private_key: ed25519.Ed25519PrivateKey) -> str:
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")


def load_private_key(value: str | bytes | None = None, *, env_var: str | None = None, file_path: Path | None = None) -> ed25519.Ed25519PrivateKey:
    raw: bytes | None = None
    if file_path is not None:
        raw = file_path.read_bytes()
    elif env_var is not None:
        val = os.environ.get(env_var)
        if not val:
            raise ValueError(f"Environment variable '{env_var}' is empty or not set")
        raw = val.strip().encode("utf-8")
    elif value is not None:
        raw = value if isinstance(value, bytes) else value.strip().encode("utf-8")
    else:
        # Default to checking MODEL_RELEASE_SIGNING_KEY env var
        val = os.environ.get("MODEL_RELEASE_SIGNING_KEY")
        if val:
            raw = val.strip().encode("utf-8")
        else:
            raise ValueError("No private key provided and MODEL_RELEASE_SIGNING_KEY is not set")

    # Try PEM format first
    if b"BEGIN" in raw and b"PRIVATE KEY" in raw:
        loaded = serialization.load_pem_private_key(raw, password=None)
        if not isinstance(loaded, ed25519.Ed25519PrivateKey):
            raise TypeError("Expected Ed25519 private key")
        return loaded

    # Try hex format (64 chars = 32 bytes)
    stripped = raw.strip()
    if len(stripped) == 64:
        try:
            return ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(stripped.decode("ascii")))
        except ValueError:
            pass

    # Try base64 format (44 chars = 32 bytes)
    try:
        decoded = base64.b64decode(stripped)
        if len(decoded) == 32:
            return ed25519.Ed25519PrivateKey.from_private_bytes(decoded)
    except (ValueError, binascii.Error):
        pass

    raise ValueError("Unable to parse private key: expected PKCS#8 PEM, 64-character hex, or base64 32-byte seed")


def load_public_key(value: str | bytes | Path | ed25519.Ed25519PublicKey | None = None) -> ed25519.Ed25519PublicKey:
    if isinstance(value, ed25519.Ed25519PublicKey):
        return value

    raw: bytes
    if value is None:
        if not DEFAULT_PUBLIC_KEY_PATH.exists():
            raise FileNotFoundError(f"Default public key file not found: {DEFAULT_PUBLIC_KEY_PATH}")
        raw = DEFAULT_PUBLIC_KEY_PATH.read_bytes()
    elif isinstance(value, Path):
        raw = value.read_bytes()
    elif isinstance(value, str):
        raw = value.strip().encode("utf-8")
    else:
        raw = value

    # Try PEM
    if b"BEGIN" in raw and b"PUBLIC KEY" in raw:
        loaded = serialization.load_pem_public_key(raw)
        if not isinstance(loaded, ed25519.Ed25519PublicKey):
            raise TypeError("Expected Ed25519 public key")
        return loaded

    # Try hex format (64 chars = 32 bytes)
    stripped = raw.strip()
    if len(stripped) == 64:
        try:
            return ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(stripped.decode("ascii")))
        except ValueError:
            pass

    # Try base64 format
    try:
        decoded = base64.b64decode(stripped)
        if len(decoded) == 32:
            return ed25519.Ed25519PublicKey.from_public_bytes(decoded)
    except (ValueError, binascii.Error):
        pass

    raise ValueError("Unable to parse public key: expected SubjectPublicKeyInfo PEM, 64-character hex, or base64 32-byte key")


def create_signature_payload(
    target_path: Path,
    private_key: ed25519.Ed25519PrivateKey,
    *,
    key_id: str = DEFAULT_KEY_ID,
) -> dict[str, Any]:
    content = target_path.read_bytes()
    target_sha256 = hashlib.sha256(content).hexdigest()
    signature_bytes = private_key.sign(content)
    signature_b64 = base64.b64encode(signature_bytes).decode("ascii")

    return {
        "schemaVersion": 1,
        "algorithm": "ed25519",
        "keyId": key_id,
        "targetFile": target_path.name,
        "targetSha256": target_sha256,
        "signature": signature_b64,
    }


def sign_file(
    target_path: Path,
    private_key: ed25519.Ed25519PrivateKey,
    *,
    key_id: str = DEFAULT_KEY_ID,
    sig_path: Path | None = None,
) -> Path:
    if sig_path is None:
        sig_path = target_path.with_name(target_path.name + ".sig")

    payload = create_signature_payload(target_path, private_key, key_id=key_id)
    sig_path.parent.mkdir(parents=True, exist_ok=True)
    sig_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return sig_path


def verify_file(
    target_path: Path,
    sig_path: Path | None = None,
    public_key: ed25519.Ed25519PublicKey | Path | str | None = None,
) -> tuple[bool, str]:
    if not target_path.exists():
        return False, f"Target file not found: {target_path}"

    if sig_path is None:
        sig_path = target_path.with_name(target_path.name + ".sig")

    if not sig_path.exists():
        return False, f"Signature file not found: {sig_path}"

    try:
        payload = json.loads(sig_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"Malformed signature file JSON: {exc}"

    if payload.get("algorithm") != "ed25519":
        return False, f"Unsupported signature algorithm: {payload.get('algorithm')}"

    if payload.get("schemaVersion") != 1:
        return False, f"Unsupported signature schema version: {payload.get('schemaVersion')}"

    # Verify SHA-256 of target file
    content = target_path.read_bytes()
    expected_sha256 = hashlib.sha256(content).hexdigest()
    if payload.get("targetSha256") != expected_sha256:
        return False, (
            f"Digest mismatch for {target_path.name}: "
            f"expected {payload.get('targetSha256')}, got {expected_sha256}"
        )

    # Verify cryptographic signature
    try:
        sig_bytes = base64.b64decode(payload["signature"])
    except (binascii.Error, ValueError, KeyError) as exc:
        return False, f"Malformed base64 signature: {exc}"

    try:
        pub = load_public_key(public_key)
        pub.verify(sig_bytes, content)
    except InvalidSignature:
        return False, "Cryptographic signature verification failed: signature does not match target bytes"
    except (ValueError, TypeError, OSError) as exc:
        return False, f"Public key verification error: {exc}"

    return True, "Signature is valid"
