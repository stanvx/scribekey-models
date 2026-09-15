from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KEY_ID = "scribekey-release-2026"
DEFAULT_ALGORITHM = "SHA256withECDSA"
SUPPORTED_ALGORITHMS = {"SHA256withECDSA", "ecdsa-p256-sha256"}
DEFAULT_PUBLIC_KEY_PATH = ROOT / "keys" / "release-signing.pub"


def generate_keypair() -> tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]:
    """Generate a P-256 (SECP256R1) ECDSA keypair."""
    private_key = ec.generate_private_key(ec.SECP256R1())
    return private_key, private_key.public_key()


def export_public_key_pem(public_key: ec.EllipticCurvePublicKey) -> str:
    """Export public key as SubjectPublicKeyInfo PEM string."""
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")


def export_private_key_pem(private_key: ec.EllipticCurvePrivateKey) -> str:
    """Export private key as unencrypted PKCS#8 PEM string."""
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")


def load_private_key(
    value: str | bytes | None = None,
    *,
    env_var: str | None = None,
    file_path: Path | None = None,
) -> ec.EllipticCurvePrivateKey:
    """Load a P-256 ECDSA private key from PEM bytes, env var, or file."""
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

    # Load PEM format (supports PKCS#8 'PRIVATE KEY' or SEC1 'EC PRIVATE KEY')
    if b"BEGIN" in raw and b"PRIVATE KEY" in raw:
        loaded = serialization.load_pem_private_key(raw, password=None)
        if not isinstance(loaded, ec.EllipticCurvePrivateKey):
            raise TypeError(f"Expected EC private key, got {type(loaded).__name__}")
        if not isinstance(loaded.curve, ec.SECP256R1):
            raise TypeError(f"Expected SECP256R1 (P-256) curve, got {loaded.curve.name}")
        return loaded

    raise ValueError("Unable to parse private key: expected PKCS#8 or SEC1 PEM format for P-256 EC key")


def load_public_key(
    value: str | bytes | Path | ec.EllipticCurvePublicKey | None = None,
) -> ec.EllipticCurvePublicKey:
    """Load a P-256 ECDSA public key from SubjectPublicKeyInfo PEM or existing instance."""
    if isinstance(value, ec.EllipticCurvePublicKey):
        if not isinstance(value.curve, ec.SECP256R1):
            raise TypeError(f"Expected SECP256R1 (P-256) curve, got {value.curve.name}")
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

    # Load PEM format
    if b"BEGIN" in raw and b"PUBLIC KEY" in raw:
        loaded = serialization.load_pem_public_key(raw)
        if not isinstance(loaded, ec.EllipticCurvePublicKey):
            raise TypeError(f"Expected EC public key, got {type(loaded).__name__}")
        if not isinstance(loaded.curve, ec.SECP256R1):
            raise TypeError(f"Expected SECP256R1 (P-256) curve, got {loaded.curve.name}")
        return loaded

    raise ValueError("Unable to parse public key: expected SubjectPublicKeyInfo PEM format for P-256 EC key")


def create_signature_payload(
    target_path: Path,
    private_key: ec.EllipticCurvePrivateKey,
    *,
    key_id: str = DEFAULT_KEY_ID,
    algorithm: str = DEFAULT_ALGORITHM,
) -> dict[str, Any]:
    """Create a detached signature payload over the EXACT bytes of target_path using P-256 / SHA256withECDSA."""
    content = target_path.read_bytes()
    target_sha256 = hashlib.sha256(content).hexdigest()
    signature_bytes = private_key.sign(content, ec.ECDSA(hashes.SHA256()))
    signature_b64 = base64.b64encode(signature_bytes).decode("ascii")

    return {
        "schemaVersion": 1,
        "algorithm": algorithm,
        "keyId": key_id,
        "targetFile": target_path.name,
        "targetSha256": target_sha256,
        "signature": signature_b64,
    }


def sign_file(
    target_path: Path,
    private_key: ec.EllipticCurvePrivateKey,
    *,
    key_id: str = DEFAULT_KEY_ID,
    algorithm: str = DEFAULT_ALGORITHM,
    sig_path: Path | None = None,
) -> Path:
    """Sign target_path and write detached signature to sig_path (defaults to <target>.sig)."""
    if sig_path is None:
        sig_path = target_path.with_name(target_path.name + ".sig")

    payload = create_signature_payload(
        target_path,
        private_key,
        key_id=key_id,
        algorithm=algorithm,
    )
    sig_path.parent.mkdir(parents=True, exist_ok=True)
    sig_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return sig_path


def verify_file(
    target_path: Path,
    sig_path: Path | None = None,
    public_key: ec.EllipticCurvePublicKey | Path | str | None = None,
) -> tuple[bool, str]:
    """Verify a detached ECDSA signature over the EXACT bytes of target_path."""
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

    algorithm = payload.get("algorithm")
    if algorithm not in SUPPORTED_ALGORITHMS:
        return False, f"Unsupported signature algorithm: {algorithm}"

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

    # Decode signature bytes
    try:
        sig_bytes = base64.b64decode(payload["signature"])
    except (binascii.Error, ValueError, KeyError) as exc:
        return False, f"Malformed base64 signature: {exc}"

    # Verify cryptographic signature against exact bytes
    try:
        pub = load_public_key(public_key)
        pub.verify(sig_bytes, content, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:
        return False, "Cryptographic signature verification failed: signature does not match target bytes"
    except (ValueError, TypeError, OSError) as exc:
        return False, f"Public key verification error: {exc}"

    return True, "Signature is valid"
