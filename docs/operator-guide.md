# Release Authority Operator Guide

This document describes the publication authority, promotion workflow, and trust contract for ScribeKey model catalogues.

## Architecture Overview

1. **Canonical Metadata (`catalog/`)**:
   Human-edited specifications describing supported speech, cleanup, and diarization models. All model asset URLs must pin immutable revisions (e.g. 40-character commit hashes on Hugging Face; no moving references like `main` or `latest`).
2. **Deterministic Outputs (`generated/`)**:
   Derived JSON catalogues and channel manifests. Generated outputs are strictly deterministic and reproducible.
3. **Immutable Releases (`catalog/releases/`, `generated/releases/`)**:
   Releases freeze qualified catalogue snapshots (`model_catalog.json`, `speaker_diarization_manifest.json`, `cleanup_model_catalog.json`) under an immutable version tag (e.g. `2026.09.1`).
4. **Channel Pointers (`catalog/channels.yaml`, `generated/channels/`)**:
   The `qa` and `stable` channels are pointers referencing immutable releases by release ID and content digests. They do NOT duplicate mutable artifact definitions.
   Channel manifests reference catalogs via relative URI paths (`../releases/<release-id>/<filename>`).

## Authoritative Trust Contract

- **Algorithm**: P-256 (SECP256R1) with `SHA256withECDSA` detached signature over the EXACT bytes of the generated channel document.
- **Anti-Rollback Sequence**: Every channel manifest contains a strictly monotonically increasing integer `sequence` and an `issuedAt` ISO-8601 timestamp.
  - Lower sequence numbers are rejected by consumers.
  - Modifying publication metadata or target release at the same sequence is rejected.
  - Re-running an identical promotion at the current sequence is an idempotent no-op (does not re-sign or change `issuedAt`).
- **Emergency Rollback**: Performed by promoting the channel to an older, known-good release ID with a **NEW, HIGHER sequence number**. Consumers accept the higher sequence and rollback safely to the older immutable snapshot.
- **No Hard Expiry**: Channels and catalogues do not hard-expire.
- **Compatibility Metadata**: Channels require compatibility constraints consumed by the Android runtime:
  ```yaml
  compatibility:
    minAndroidApiLevel: 28
    minAppVersionCode: 10
    catalogsSchemaVersion: 1
    supportedRuntimeFamilies: ["sherpa-onnx", "gguf", "pyannote"]
    supportedConfigFamilies: ["speech-model-catalog", "cleanup-model-catalog", "speaker-diarization-manifest"]
  ```
- **Android Rejection Contract**: Android consumers reject unknown runtime/config families and any payload with executable extensions (`.apk`, `.dex`, `.so`, `.jar`, `.class`, `.sh`, etc.).

## Operator Workflows

### 1. Creating an Immutable Release

When models in `catalog/` are qualified:

```bash
scribekey-models release create --id 2026.09.1 --description "Release description"
```

This freezes the catalog snapshots under `generated/releases/2026.09.1/` and records their cryptographic digests in `catalog/releases/2026.09.1.yaml`.

### 2. Reviewable Promotion via GitHub Actions

1. In GitHub Actions, navigate to **Prepare Release Promotion** (`workflow_dispatch`).
2. Select target channel (`qa` or `stable`), enter `release` ID (`2026.09.1`), and optional notes.
3. The workflow runs with the secret `MODEL_RELEASE_SIGNING_KEY`:
   - Validates release definitions and guardrails.
   - Stages promotion and signs the channel manifest in temporary isolation.
   - Fails closed immediately if `MODEL_RELEASE_SIGNING_KEY` is not configured.
   - Opens a Pull Request from branch `promote/<channel>-<release>`.
4. **Atomic Publication Boundary**: A human must review the PR diff. Merging the PR into `main` atomically publishes the verified distribution metadata to clients. Auto-merge is strictly disabled.

### 3. Emergency Rollback

To roll back a channel (e.g. `stable`) to an earlier known-good release `2026.08.1`:

1. Trigger the promotion workflow or CLI with `--channel stable --release 2026.08.1`.
2. The tooling automatically assigns a new, higher sequence number (e.g. sequence 3).
3. The generated `stable.json` points to `../releases/2026.08.1/...`.
4. Review and merge the rollback PR.

### 4. Local CLI Usage

```bash
# Validate catalogues, schemas, guardrails, and signatures
scribekey-models validate

# Check freshness of generated files
scribekey-models generate --check

# Promote a release (requires signing key)
scribekey-models promote --channel qa --release 2026.09.1 --key-env MODEL_RELEASE_SIGNING_KEY
```

## Production Key Provisioning Status

The production P-256 (SECP256R1) signing key is provisioned for this release authority:

- the private PKCS#8 PEM is stored only in the repository's GitHub Actions secret named `MODEL_RELEASE_SIGNING_KEY`
- the matching SubjectPublicKeyInfo PEM public key is committed at `keys/release-signing.pub`
- the Android application embeds the same public key as its catalogue trust root
- committed QA, stable, and release-manifest signatures are produced from that production key

The local plaintext private-key file used during provisioning is not retained. Future promotions use the GitHub Actions secret. Key rotation requires publishing the replacement public key to Android before channels are signed exclusively by the replacement key.

## Redistribution Clearance & Mirroring

- Models must not be asserted as `cleared: true` in `recoveryAssets` without explicit, verified legal clearance and attribution.
- Current production outputs leave `recoveryAssets` empty until explicit clearance evidence exists.
- Mirroring tooling (`scribekey-models mirror plan`) operates in dry-run mode and never uploads model binary files to Git history.
