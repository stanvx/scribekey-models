from __future__ import annotations

import base64
import datetime
import http.client
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

REDIRECT_CODES = {301, 302, 303, 307, 308}
HEAD_UNSUPPORTED_CODES = {405, 501}
HEX64_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
CONTENT_RANGE_RE = re.compile(r"bytes\s+\d+-\d+/(\d+|\*)", re.IGNORECASE)
DEFAULT_USER_AGENT = "scribekey-models-source-health/1"


@dataclass(frozen=True)
class SourceTarget:
    catalog: str
    model_id: str
    file_name: str
    source_index: int
    url: str
    expected_size_bytes: int
    expected_sha256: str


@dataclass(frozen=True)
class SourceMetadata:
    status_code: int
    size_bytes: int | None
    sha256: str | None


@dataclass(frozen=True)
class SourceCheck:
    target: SourceTarget
    healthy: bool
    reason: str
    observed_size_bytes: int | None = None
    checksum_verified: bool = False


@dataclass(frozen=True)
class HealthReport:
    channel: str
    release_id: str
    checked_at: str
    checks: tuple[SourceCheck, ...]

    @property
    def healthy(self) -> bool:
        return all(check.healthy for check in self.checks)

    @property
    def failures(self) -> tuple[SourceCheck, ...]:
        return tuple(check for check in self.checks if not check.healthy)

    @property
    def incident_key(self) -> str:
        return f"scribekey-model-source-health:{self.channel}:{self.release_id}"

    @property
    def incident_marker(self) -> str:
        return f"<!-- {self.incident_key} -->"

    @property
    def incident_title(self) -> str:
        return f"Model source health incident: {self.channel} / {self.release_id}"

    def as_json(self) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "channel": self.channel,
            "releaseId": self.release_id,
            "checkedAt": self.checked_at,
            "healthy": self.healthy,
            "targetCount": len(self.checks),
            "failureCount": len(self.failures),
            "incidentKey": self.incident_key,
            "incidentMarker": self.incident_marker,
            "incidentTitle": self.incident_title,
            "incidentBody": self.incident_body(),
            "checks": [
                {
                    "target": asdict(check.target),
                    "healthy": check.healthy,
                    "reason": check.reason,
                    "observedSizeBytes": check.observed_size_bytes,
                    "checksumVerified": check.checksum_verified,
                }
                for check in self.checks
            ],
        }

    def incident_body(self) -> str:
        lines = [
            self.incident_marker,
            "## Model source health failure",
            "",
            f"Distribution channel: `{self.channel}`",
            f"Release: `{self.release_id}`",
            f"Last checked: `{self.checked_at}`",
            f"Failed sources: **{len(self.failures)} / {len(self.checks)}**",
            "",
            "The catalogue was **not** modified. This issue tracks source availability or immutable metadata drift.",
            "",
            "| Model | File | Source | Problem |",
            "| --- | --- | ---: | --- |",
        ]
        for check in self.failures:
            target = check.target
            reason = check.reason.replace("|", "\\|")
            lines.append(
                f"| `{target.model_id}` | `{target.file_name}` | {target.source_index + 1} | {reason} |"
            )
        lines.extend(
            [
                "",
                "Resolve the source or publish reviewed replacement metadata through the normal release/promotion flow.",
                "Do not mutate an existing immutable release snapshot in place.",
                "",
            ]
        )
        return "\n".join(lines)


def load_channel_targets(root: Path, channel: str = "stable") -> tuple[str, list[SourceTarget]]:
    channel_path = root / "generated" / "channels" / f"{channel}.json"
    channel_data = _load_json(channel_path)
    release_id = str(channel_data.get("releaseId", "")).strip()
    if not release_id:
        raise ValueError(f"Channel '{channel}' has no releaseId")

    catalogs = channel_data.get("catalogs")
    if not isinstance(catalogs, dict):
        raise TypeError(f"Channel '{channel}' has no catalog map")

    targets: list[SourceTarget] = []
    for catalog_name in ("speech", "cleanup", "diarization"):
        catalog_ref = catalogs.get(catalog_name)
        if not isinstance(catalog_ref, dict):
            raise TypeError(f"Channel '{channel}' is missing the {catalog_name} catalog reference")
        path = _resolve_catalog_path(root, channel_path, str(catalog_ref.get("path", "")), release_id)
        data = _load_json(path)
        catalog_targets = _targets_from_catalog(catalog_name, data)
        if not catalog_targets:
            raise ValueError(f"Published {catalog_name} catalog contains no model artifact sources")
        targets.extend(catalog_targets)
    if not targets:
        raise ValueError(f"Channel '{channel}' contains no model artifact sources")
    return release_id, targets


def check_channel_sources(
    root: Path,
    channel: str = "stable",
    *,
    timeout_seconds: float = 20.0,
    probe: Callable[[str, float], SourceMetadata] | None = None,
    now: datetime.datetime | None = None,
) -> HealthReport:
    release_id, targets = load_channel_targets(root, channel)
    probe_fn = probe or probe_source_metadata
    checks = tuple(_check_target(target, probe_fn, timeout_seconds) for target in targets)
    checked = (now or datetime.datetime.now(datetime.UTC)).astimezone(datetime.UTC)
    return HealthReport(
        channel=channel,
        release_id=release_id,
        checked_at=checked.strftime("%Y-%m-%dT%H:%M:%SZ"),
        checks=checks,
    )


def write_report(report: HealthReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_json(), indent=2) + "\n", encoding="utf-8")


def probe_source_metadata(url: str, timeout_seconds: float) -> SourceMetadata:
    initial = _request_without_redirects(url, timeout_seconds)
    size = _size_from_headers(initial[1], accept_content_length=initial[0] not in REDIRECT_CODES)
    checksum = _sha256_from_headers(initial[1])

    if initial[0] in REDIRECT_CODES or initial[0] in HEAD_UNSUPPORTED_CODES or size is None:
        final_status, final_headers = _range_probe(url, timeout_seconds)
        final_size = _size_from_headers(final_headers, accept_content_length=True)
        size = final_size if final_size is not None else size
        checksum = checksum or _sha256_from_headers(final_headers)
        return SourceMetadata(final_status, size, checksum)

    return SourceMetadata(initial[0], size, checksum)


def _check_target(
    target: SourceTarget,
    probe: Callable[[str, float], SourceMetadata],
    timeout_seconds: float,
) -> SourceCheck:
    try:
        metadata = probe(target.url, timeout_seconds)
    except (OSError, TimeoutError, ValueError, http.client.HTTPException, urllib.error.URLError) as error:
        return SourceCheck(target, False, f"source unavailable: {error}")

    if metadata.status_code not in {200, 206}:
        return SourceCheck(target, False, f"unexpected HTTP status {metadata.status_code}")
    if metadata.size_bytes is None and metadata.sha256 is None:
        return SourceCheck(target, False, "source returned no verifiable artifact metadata")

    if metadata.size_bytes is not None and metadata.size_bytes != target.expected_size_bytes:
        return SourceCheck(
            target,
            False,
            f"size mismatch: expected {target.expected_size_bytes}, got {metadata.size_bytes}",
            observed_size_bytes=metadata.size_bytes,
        )

    if metadata.sha256 is not None and metadata.sha256.lower() != target.expected_sha256.lower():
        return SourceCheck(
            target,
            False,
            f"SHA-256 mismatch: expected {target.expected_sha256}, got {metadata.sha256}",
            observed_size_bytes=metadata.size_bytes,
            checksum_verified=False,
        )

    return SourceCheck(
        target,
        True,
        "ok",
        observed_size_bytes=metadata.size_bytes,
        checksum_verified=metadata.sha256 is not None,
    )


def _targets_from_catalog(catalog: str, data: dict[str, object]) -> list[SourceTarget]:
    if catalog == "speech":
        entries = data.get("models", [])
        return _speech_targets(entries if isinstance(entries, list) else [])
    if catalog == "cleanup":
        production = data.get("production")
        return _cleanup_targets(production if isinstance(production, dict) else None)
    if catalog == "diarization":
        entries = data.get("models", [])
        return _diarization_targets(entries if isinstance(entries, list) else [])
    return []


def _speech_targets(entries: list[object]) -> list[SourceTarget]:
    targets: list[SourceTarget] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id", "unknown"))
        files = entry.get("files", [])
        if not isinstance(files, list):
            continue
        for file_data in files:
            if isinstance(file_data, dict):
                targets.extend(_file_targets("speech", model_id, file_data, "name"))
    return targets


def _cleanup_targets(production: dict[str, object] | None) -> list[SourceTarget]:
    if production is None:
        return []
    return _file_targets(
        "cleanup",
        str(production.get("modelId", "unknown")),
        production,
        "bundleFileName",
    )


def _diarization_targets(entries: list[object]) -> list[SourceTarget]:
    targets: list[SourceTarget] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        targets.extend(
            _file_targets("diarization", str(entry.get("role", "unknown")), entry, "fileName")
        )
    return targets


def _file_targets(
    catalog: str,
    model_id: str,
    file_data: Mapping[str, object],
    name_key: str,
) -> list[SourceTarget]:
    primary = str(file_data.get("downloadUrl", "")).strip()
    fallbacks = file_data.get("downloadUrls", [])
    urls = [primary]
    if isinstance(fallbacks, list):
        urls.extend(str(url).strip() for url in fallbacks)
    urls = [url for index, url in enumerate(urls) if url and url not in urls[:index]]
    expected_size = int(file_data.get("sizeBytes", 0))
    expected_sha = str(file_data.get("sha256", "")).lower()
    return [
        SourceTarget(
            catalog=catalog,
            model_id=model_id,
            file_name=str(file_data.get(name_key, "unknown")),
            source_index=index,
            url=url,
            expected_size_bytes=expected_size,
            expected_sha256=expected_sha,
        )
        for index, url in enumerate(urls)
    ]


def _resolve_catalog_path(root: Path, channel_path: Path, raw_path: str, release_id: str) -> Path:
    if not raw_path:
        raise ValueError("Channel catalog path is blank")
    candidate = (channel_path.parent / raw_path).resolve()
    expected_root = (root / "generated" / "releases" / release_id).resolve()
    if candidate.parent != expected_root:
        raise ValueError(f"Channel catalog path escapes immutable release '{release_id}': {raw_path}")
    return candidate


def _request_without_redirects(url: str, timeout_seconds: float) -> tuple[int, Mapping[str, str]]:
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": DEFAULT_USER_AGENT, "Accept-Encoding": "identity"},
    )
    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            return response.status, dict(response.headers.items())
    except urllib.error.HTTPError as error:
        if error.code in REDIRECT_CODES or error.code in HEAD_UNSUPPORTED_CODES:
            return error.code, dict(error.headers.items())
        raise


def _range_probe(url: str, timeout_seconds: float) -> tuple[int, Mapping[str, str]]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept-Encoding": "identity",
            "Range": "bytes=0-0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response.read(1)
        return response.status, dict(response.headers.items())


def _size_from_headers(headers: Mapping[str, str], *, accept_content_length: bool) -> int | None:
    lowered = {key.lower(): value for key, value in headers.items()}
    linked_size = lowered.get("x-linked-size")
    if linked_size and linked_size.isdigit():
        return int(linked_size)
    content_range = lowered.get("content-range", "")
    match = CONTENT_RANGE_RE.fullmatch(content_range.strip())
    if match and match.group(1) != "*":
        return int(match.group(1))
    content_length = lowered.get("content-length")
    if accept_content_length and content_length and content_length.isdigit():
        return int(content_length)
    return None


def _sha256_from_headers(headers: Mapping[str, str]) -> str | None:
    lowered = {key.lower(): value.strip() for key, value in headers.items()}
    for key in ("x-linked-etag", "x-checksum-sha256", "x-amz-meta-sha256"):
        value = lowered.get(key, "").strip('"').removeprefix("W/").strip('"')
        if HEX64_RE.fullmatch(value):
            return value.lower()

    digest = lowered.get("digest", "")
    for part in digest.split(","):
        algorithm, separator, encoded = part.strip().partition("=")
        if separator and algorithm.lower() in {"sha-256", "sha256"}:
            try:
                decoded = base64.b64decode(encoded, validate=True).hex()
            except ValueError:
                continue
            if HEX64_RE.fullmatch(decoded):
                return decoded.lower()

    return None


def _load_json(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return data
