from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scribekey_models.signing import verify_file

ROOT = Path(__file__).resolve().parents[2]
CATALOG_DIR = ROOT / "catalog"
RELEASES_DIR = CATALOG_DIR / "releases"
CHANNELS_FILE = CATALOG_DIR / "channels.yaml"
GENERATED_DIR = ROOT / "generated"

HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
MUTABLE_REFS = {"main", "master", "head", "latest", "trunk", "dev", "develop"}
BINARY_EXTENSIONS = {
    ".onnx",
    ".bin",
    ".safetensors",
    ".pt",
    ".pth",
    ".gguf",
    ".tflite",
    ".ckpt",
    ".tar.gz",
    ".zip",
    ".h5",
}
EXECUTABLE_PAYLOAD_EXTENSIONS = {
    ".apk",
    ".dex",
    ".so",
    ".jar",
    ".class",
    ".sh",
    ".bash",
    ".exe",
    ".elf",
    ".bat",
    ".cmd",
    ".ps1",
    ".dylib",
    ".dll",
}
KNOWN_RUNTIME_FAMILIES = {"sherpa-onnx", "gguf", "pyannote"}
KNOWN_CONFIG_FAMILIES = {
    "speech-model-catalog",
    "cleanup-model-catalog",
    "speaker-diarization-manifest",
}
PERMISSIVE_LICENSES = {
    "MIT",
    "Apache-2.0",
    "BSD-3-Clause",
    "BSD-2-Clause",
    "ISC",
    "CC0-1.0",
    "Unlicense",
}
MAX_TRACKED_FILE_BYTES = 2 * 1024 * 1024  # 2MB maximum for any metadata/tooling file


@dataclass(frozen=True)
class GuardrailIssue:
    source: str
    message: str


def validate_download_url(source: str, url: str, model_id: str) -> list[GuardrailIssue]:
    issues: list[GuardrailIssue] = []
    if not url:
        return [GuardrailIssue(source, f"Model '{model_id}' has empty downloadUrl")]

    for ref in MUTABLE_REFS:
        if f"/resolve/{ref}/" in url or f"/{ref}/" in url:
            issues.append(
                GuardrailIssue(
                    source,
                    f"Model '{model_id}' URL uses mutable ref '{ref}': {url}",
                )
            )

    if "huggingface.co" in url and "/resolve/" in url:
        parts = url.split("/resolve/")[1].split("/")
        revision = parts[0]
        if not HEX40_RE.match(revision):
            issues.append(
                GuardrailIssue(
                    source,
                    f"Model '{model_id}' HuggingFace URL must use a 40-character commit hash, got '{revision}'",
                )
            )

    return issues


def validate_immutable_source_refs(
    speech_data: dict[str, Any],
    cleanup_data: dict[str, Any],
    diarization_data: dict[str, Any],
) -> list[GuardrailIssue]:
    issues: list[GuardrailIssue] = []

    # Check speech models
    for model in speech_data.get("models", []):
        model_id = model.get("id", "unknown")
        for f in model.get("files", []):
            url = f.get("downloadUrl", "")
            issues.extend(validate_download_url("catalog/speech.yaml", url, model_id))

    # Check cleanup models
    prod = cleanup_data.get("production")
    if prod:
        model_id = prod.get("modelId", "production")
        rev = prod.get("revision", "")
        if rev.lower() in MUTABLE_REFS:
            issues.append(
                GuardrailIssue("catalog/cleanup.yaml", f"Cleanup model '{model_id}' revision cannot be mutable ref '{rev}'")
            )
        url = prod.get("downloadUrl", "")
        issues.extend(validate_download_url("catalog/cleanup.yaml", url, model_id))

    for cand in cleanup_data.get("candidates", []):
        model_id = cand.get("modelId", "candidate")
        rev = cand.get("revision", "")
        if rev.lower() in MUTABLE_REFS:
            issues.append(
                GuardrailIssue("catalog/cleanup.yaml", f"Cleanup candidate '{model_id}' revision cannot be mutable ref '{rev}'")
            )
        url = cand.get("downloadUrl", "")
        issues.extend(validate_download_url("catalog/cleanup.yaml", url, model_id))

    # Check diarization models
    for model in diarization_data.get("models", []):
        role = model.get("role", "unknown")
        rev = model.get("sourceRevision", "")
        if rev.lower() in MUTABLE_REFS:
            issues.append(
                GuardrailIssue("catalog/diarization.yaml", f"Diarization model '{role}' sourceRevision cannot be mutable ref '{rev}'")
            )
        url = model.get("downloadUrl", "")
        issues.extend(validate_download_url("catalog/diarization.yaml", url, role))

    return issues


def validate_identities_and_integrity(
    speech_data: dict[str, Any],
    cleanup_data: dict[str, Any],
    diarization_data: dict[str, Any],
) -> list[GuardrailIssue]:
    issues: list[GuardrailIssue] = []

    # Speech IDs and digests
    seen_speech_ids: set[str] = set()
    all_speech_ids: set[str] = set()
    for model in speech_data.get("models", []):
        m_id = model.get("id")
        if not m_id:
            issues.append(GuardrailIssue("catalog/speech.yaml", "Speech model missing id"))
            continue
        if m_id in seen_speech_ids:
            issues.append(GuardrailIssue("catalog/speech.yaml", f"Duplicate speech model id: '{m_id}'"))
        seen_speech_ids.add(m_id)
        all_speech_ids.add(m_id)

        for f in model.get("files", []):
            sha = f.get("sha256", "")
            if not HEX64_RE.match(sha):
                issues.append(GuardrailIssue("catalog/speech.yaml", f"Model '{m_id}' invalid sha256: '{sha}'"))
            size = f.get("sizeBytes")
            if not isinstance(size, int) or size <= 0:
                issues.append(GuardrailIssue("catalog/speech.yaml", f"Model '{m_id}' invalid sizeBytes: '{size}'"))

    for model in speech_data.get("models", []):
        repl = model.get("replacementId")
        if repl and repl not in all_speech_ids:
            issues.append(GuardrailIssue("catalog/speech.yaml", f"Model '{model.get('id')}' replacementId '{repl}' not found in catalogue"))

    # Diarization digests
    for model in diarization_data.get("models", []):
        role = model.get("role", "unknown")
        sha = model.get("sha256", "")
        if not HEX64_RE.match(sha):
            issues.append(GuardrailIssue("catalog/diarization.yaml", f"Diarization model '{role}' invalid sha256: '{sha}'"))
        size = model.get("sizeBytes")
        if not isinstance(size, int) or size <= 0:
            issues.append(GuardrailIssue("catalog/diarization.yaml", f"Diarization model '{role}' invalid sizeBytes: '{size}'"))

    # Cleanup digests
    all_cleanup = []
    if cleanup_data.get("production"):
        all_cleanup.append(cleanup_data["production"])
    all_cleanup.extend(cleanup_data.get("candidates", []))

    for model in all_cleanup:
        m_id = model.get("modelId", "unknown")
        sha = model.get("sha256", "")
        if not HEX64_RE.match(sha):
            issues.append(GuardrailIssue("catalog/cleanup.yaml", f"Cleanup model '{m_id}' invalid sha256: '{sha}'"))
        size = model.get("sizeBytes")
        if not isinstance(size, int) or size <= 0:
            issues.append(GuardrailIssue("catalog/cleanup.yaml", f"Cleanup model '{m_id}' invalid sizeBytes: '{size}'"))

    return issues


def validate_no_executable_payloads(
    speech_data: dict[str, Any],
    cleanup_data: dict[str, Any],
    diarization_data: dict[str, Any],
    releases_data: dict[str, dict[str, Any]] | None = None,
) -> list[GuardrailIssue]:
    """Ensure no model payload is an executable binary or script rejected by Android."""
    issues: list[GuardrailIssue] = []

    def _check(source: str, filename: str, entity_id: str) -> None:
        ext = Path(filename).suffix.lower()
        if ext in EXECUTABLE_PAYLOAD_EXTENSIONS:
            issues.append(
                GuardrailIssue(
                    source,
                    f"Entity '{entity_id}' references executable payload '{filename}' ({ext}) "
                    f"which is rejected by Android runtime guardrails",
                )
            )

    for model in speech_data.get("models", []):
        m_id = model.get("id", "unknown")
        for f in model.get("files", []):
            _check("catalog/speech.yaml", f.get("name", ""), m_id)

    prod = cleanup_data.get("production")
    if prod:
        m_id = prod.get("modelId", "production")
        url = prod.get("downloadUrl", "")
        if url:
            _check("catalog/cleanup.yaml", Path(url).name, m_id)

    for cand in cleanup_data.get("candidates", []):
        m_id = cand.get("modelId", "candidate")
        url = cand.get("downloadUrl", "")
        if url:
            _check("catalog/cleanup.yaml", Path(url).name, m_id)

    for model in diarization_data.get("models", []):
        role = model.get("role", "unknown")
        url = model.get("downloadUrl", "")
        if url:
            _check("catalog/diarization.yaml", Path(url).name, role)

    if releases_data:
        for rel_id, rel_data in releases_data.items():
            recovery = rel_data.get("recoveryAssets", {})
            for model in recovery.get("clearedModels", []):
                m_id = model.get("modelId", "unknown")
                for f in model.get("files", []):
                    _check(f"catalog/releases/{rel_id}.yaml", f.get("name", ""), m_id)

    return issues


def validate_release_safety(root: Path = ROOT) -> list[GuardrailIssue]:
    issues: list[GuardrailIssue] = []

    # Excluded directories
    excluded_dirs = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "fixtures"}

    for path in root.rglob("*"):
        if any(part in excluded_dirs for part in path.parts):
            continue
        if not path.is_file():
            continue

        # Check binary extensions
        if path.suffix.lower() in BINARY_EXTENSIONS:
            issues.append(GuardrailIssue(str(path.relative_to(root)), f"Forbidden model binary file tracked in repo: {path.name}"))

        # Check file size limit
        try:
            stat = path.stat()
            if stat.st_size > MAX_TRACKED_FILE_BYTES:
                issues.append(GuardrailIssue(str(path.relative_to(root)), f"File exceeds maximum allowed size ({stat.st_size} > {MAX_TRACKED_FILE_BYTES} bytes)"))
        except OSError:
            pass

        # Check for accidentally committed secrets or tokens (only in non-code text/config files)
        if path.suffix in {".yaml", ".yml", ".json", ".md", ".txt", ".toml", ".env", ".key"}:
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                pem_markers = [
                    "-----" + "BEGIN " + "PRIVATE KEY-----",
                    "-----" + "BEGIN " + "OPENSSH PRIVATE KEY-----",
                    "-----" + "BEGIN " + "EC PRIVATE KEY-----",
                    "-----" + "BEGIN " + "RSA PRIVATE KEY-----",
                ]
                if any(marker in content for marker in pem_markers):
                    issues.append(GuardrailIssue(str(path.relative_to(root)), "Private signing key found in file! Private keys must only live in CI secrets."))
                # Detect HuggingFace user tokens: hf_ followed by 34 alphanumeric chars
                if re.search(r"\bhf_[A-Za-z0-9]{34}\b", content):
                    issues.append(GuardrailIssue(str(path.relative_to(root)), "Potential Hugging Face authentication token found! Tokens must not be committed."))
                # Detect GitHub PATs: ghp_ followed by 36 alphanumeric chars
                if re.search(r"\bghp_[A-Za-z0-9]{36}\b", content):
                    issues.append(GuardrailIssue(str(path.relative_to(root)), "Potential GitHub Personal Access Token found! Tokens must not be committed."))
            except (OSError, UnicodeDecodeError):
                continue

    return issues


def validate_redistribution_clearance(release_data: dict[str, Any]) -> list[GuardrailIssue]:
    issues: list[GuardrailIssue] = []
    recovery = release_data.get("recoveryAssets")
    if not recovery:
        return issues

    for item in recovery.get("clearedModels", []):
        model_id = item.get("modelId", "unknown")
        if not item.get("cleared"):
            issues.append(GuardrailIssue("recoveryAssets", f"Model '{model_id}' is listed in recoveryAssets but cleared is false"))
        lic = item.get("licenseId", "")
        if lic not in PERMISSIVE_LICENSES:
            issues.append(
                GuardrailIssue(
                    "recoveryAssets",
                    f"Model '{model_id}' license '{lic}' is not in redistribution-cleared permissive licenses ({', '.join(sorted(PERMISSIVE_LICENSES))})",
                )
            )

        for f in item.get("files", []):
            fname = f.get("name", "")
            sha = f.get("sha256", "")
            if not HEX64_RE.match(sha):
                issues.append(GuardrailIssue("recoveryAssets", f"Model '{model_id}' file '{fname}' invalid sha256: '{sha}'"))
            url = f.get("upstreamUrl", "")
            issues.extend(validate_download_url("recoveryAssets", url, f"{model_id}:{fname}"))

    return issues


def validate_release_snapshot_integrity(
    release_id: str,
    release_data: dict[str, Any],
    generated_dir: Path = GENERATED_DIR,
) -> list[GuardrailIssue]:
    issues: list[GuardrailIssue] = []
    release_dir = generated_dir / "releases" / release_id
    for catalog_name, declared in release_data.get("catalogs", {}).items():
        filename = declared.get("filename")
        if not filename:
            issues.append(GuardrailIssue(catalog_name, "Release catalogue is missing filename"))
            continue
        snapshot = release_dir / str(filename)
        if not snapshot.is_file():
            issues.append(
                GuardrailIssue(catalog_name, f"Immutable release snapshot is missing: {snapshot}")
            )
            continue
        content = snapshot.read_bytes()
        actual_sha256 = hashlib.sha256(content).hexdigest()
        actual_size = len(content)
        if actual_sha256 != declared.get("sha256"):
            issues.append(
                GuardrailIssue(
                    catalog_name,
                    f"Immutable release snapshot digest changed for {filename}",
                )
            )
        if actual_size != declared.get("sizeBytes"):
            issues.append(
                GuardrailIssue(
                    catalog_name,
                    f"Immutable release snapshot size changed for {filename}",
                )
            )
    return issues


def validate_channel_and_release_pointers(
    channels_data: dict[str, Any],
    releases_dir: Path = RELEASES_DIR,
    generated_dir: Path = GENERATED_DIR,
) -> list[GuardrailIssue]:
    issues: list[GuardrailIssue] = []

    channels = channels_data.get("channels", {})
    for chan_name in ["qa", "stable"]:
        chan = channels.get(chan_name)
        if not chan:
            issues.append(GuardrailIssue("catalog/channels.yaml", f"Missing channel pointer for '{chan_name}'"))
            continue
        rel_id = chan.get("targetRelease")
        if not rel_id:
            issues.append(GuardrailIssue("catalog/channels.yaml", f"Channel '{chan_name}' missing targetRelease"))
            continue

        rel_file = releases_dir / f"{rel_id}.yaml"
        if not rel_file.exists():
            issues.append(
                GuardrailIssue(
                    "catalog/channels.yaml",
                    f"Channel '{chan_name}' points to non-existent release definition: {rel_file}",
                )
            )

        seq = chan.get("sequence")
        if seq is None or not isinstance(seq, int) or seq < 1:
            issues.append(
                GuardrailIssue(
                    "catalog/channels.yaml",
                    f"Channel '{chan_name}' has invalid anti-rollback sequence '{seq}' (must be integer >= 1)",
                )
            )

        issued_at = chan.get("issuedAt")
        if not issued_at:
            issues.append(
                GuardrailIssue(
                    "catalog/channels.yaml",
                    f"Channel '{chan_name}' missing required 'issuedAt' timestamp",
                )
            )

        comp = chan.get("compatibility")
        if not comp or not isinstance(comp, dict):
            issues.append(
                GuardrailIssue(
                    "catalog/channels.yaml",
                    f"Channel '{chan_name}' missing required 'compatibility' metadata",
                )
            )
        else:
            api_level = comp.get("minAndroidApiLevel")
            if not isinstance(api_level, int) or api_level < 21:
                issues.append(
                    GuardrailIssue(
                        "catalog/channels.yaml",
                        f"Channel '{chan_name}' compatibility.minAndroidApiLevel must be integer >= 21, got '{api_level}'",
                    )
                )

            app_version = comp.get("minAppVersionCode")
            if not isinstance(app_version, int) or app_version < 1:
                issues.append(
                    GuardrailIssue(
                        "catalog/channels.yaml",
                        f"Channel '{chan_name}' compatibility.minAppVersionCode must be integer >= 1, got '{app_version}'",
                    )
                )

            schema_ver = comp.get("catalogsSchemaVersion")
            if not isinstance(schema_ver, int) or schema_ver < 1:
                issues.append(
                    GuardrailIssue(
                        "catalog/channels.yaml",
                        f"Channel '{chan_name}' compatibility.catalogsSchemaVersion must be integer >= 1, got '{schema_ver}'",
                    )
                )

            families = comp.get("supportedRuntimeFamilies")
            if not isinstance(families, list) or not families:
                issues.append(
                    GuardrailIssue(
                        "catalog/channels.yaml",
                        f"Channel '{chan_name}' compatibility.supportedRuntimeFamilies must be non-empty list",
                    )
                )
            else:
                unknown = set(families) - KNOWN_RUNTIME_FAMILIES
                if unknown:
                    issues.append(
                        GuardrailIssue(
                            "catalog/channels.yaml",
                            f"Channel '{chan_name}' contains unknown runtime families: {unknown} "
                            f"(known: {KNOWN_RUNTIME_FAMILIES})",
                        )
                    )

            configs = comp.get("supportedConfigFamilies")
            if not isinstance(configs, list) or not configs:
                issues.append(
                    GuardrailIssue(
                        "catalog/channels.yaml",
                        f"Channel '{chan_name}' compatibility.supportedConfigFamilies must be non-empty list",
                    )
                )
            else:
                unknown_cfg = set(configs) - KNOWN_CONFIG_FAMILIES
                if unknown_cfg:
                    issues.append(
                        GuardrailIssue(
                            "catalog/channels.yaml",
                            f"Channel '{chan_name}' contains unknown config families: {unknown_cfg} "
                            f"(known: {KNOWN_CONFIG_FAMILIES})",
                        )
                    )

    return issues


def validate_signatures(
    generated_dir: Path = GENERATED_DIR,
    public_key_path: Path = ROOT / "keys" / "release-signing.pub",
) -> list[GuardrailIssue]:
    issues: list[GuardrailIssue] = []
    if not public_key_path.exists():
        issues.append(GuardrailIssue("keys/release-signing.pub", "Public key file not found"))
        return issues

    # Validate signatures on channels and releases
    for sig_file in generated_dir.rglob("*.sig"):
        target_name = sig_file.name[:-4] if sig_file.name.endswith(".sig") else sig_file.stem
        target_file = sig_file.with_name(target_name)
        if not target_file.exists():
            issues.append(GuardrailIssue(str(sig_file.relative_to(ROOT)), f"Orphaned signature file without target: {target_file.name}"))
            continue

        ok, msg = verify_file(target_file, sig_file, public_key=public_key_path)
        if not ok:
            issues.append(GuardrailIssue(str(sig_file.relative_to(ROOT)), f"Signature verification failed for {target_file.name}: {msg}"))

    return issues
