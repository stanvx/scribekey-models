from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from scribekey_models.guardrails import (
    HEX40_RE,
    HEX64_RE,
    MUTABLE_REFS,
    PERMISSIVE_LICENSES,
)


@dataclass(frozen=True)
class MirrorAssetPlan:
    release_id: str
    model_id: str
    filename: str
    sha256: str
    upstream_url: str
    license_id: str
    target_path: str


def plan_release_mirror(release_data: dict[str, Any]) -> list[MirrorAssetPlan]:
    """Plan redistribution-cleared mirror assets for a release without downloading or uploading weights."""
    release_id = str(release_data.get("releaseId", "unknown"))
    recovery = release_data.get("recoveryAssets")
    if not recovery:
        return []

    plans: list[MirrorAssetPlan] = []
    for model in recovery.get("clearedModels", []):
        if not model.get("cleared"):
            continue
        lic = model.get("licenseId", "")
        if lic not in PERMISSIVE_LICENSES:
            continue

        model_id = model.get("modelId", "unknown")
        for f in model.get("files", []):
            name = f.get("name", "")
            sha256 = f.get("sha256", "")
            url = f.get("upstreamUrl", "")

            # Ensure upstream URL is immutable
            for ref in MUTABLE_REFS:
                if f"/resolve/{ref}/" in url or f"/{ref}/" in url:
                    raise ValueError(f"Mirror upstream URL for '{model_id}:{name}' uses mutable ref '{ref}': {url}")

            if "huggingface.co" in url and "/resolve/" in url:
                parts = url.split("/resolve/")[1].split("/")
                if not HEX40_RE.match(parts[0]):
                    raise ValueError(
                        f"Mirror upstream URL for '{model_id}:{name}' does not pin a 40-character commit hash: {url}"
                    )

            if not HEX64_RE.match(sha256):
                raise ValueError(f"Invalid sha256 for '{model_id}:{name}': {sha256}")

            plans.append(
                MirrorAssetPlan(
                    release_id=release_id,
                    model_id=model_id,
                    filename=name,
                    sha256=sha256,
                    upstream_url=url,
                    license_id=lic,
                    target_path=f"mirrors/{release_id}/{model_id}/{name}",
                )
            )

    return plans


def check_mirror_configuration() -> dict[str, Any]:
    """Check mirror configuration and dry-run credential status without performing uploads."""
    hf_token_set = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    gh_token_set = bool(os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"))

    return {
        "configured": hf_token_set or gh_token_set,
        "huggingface_token_available": hf_token_set,
        "github_token_available": gh_token_set,
        "mode": "dry_run",
        "message": (
            "Mirroring hooks are bounded. No large model weights uploaded. "
            "Operator credentials can be supplied via CI secrets (HF_TOKEN or GITHUB_TOKEN) when needed."
        ),
    }
