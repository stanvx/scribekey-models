# ScribeKey Models

Open model catalogue and distribution tooling for ScribeKey-compatible on-device models.

This repository serves as the public release authority for model metadata consumed by ScribeKey on Android. It contains schemas, canonical model metadata, deterministic runtime catalogues, channel promotion records, and detached cryptographic signatures.

Large model weights remain strictly outside Git history.

## Repository Layout

```text
catalog/      Canonical human-edited model metadata (speech, cleanup, diarization, releases, channels)
schemas/      JSON schemas for canonical inputs, generated catalogues, and signatures
generated/    Deterministic runtime catalogues and signed channel manifests
keys/         Release signing public keys (release-signing.pub)
src/          Validation, guardrail, promotion, and signing tooling
tests/        Unit, reproducibility, and guardrail tests
docs/         Operator guide, compatibility contracts, and contribution notes
```

## Developer & Operator Quickstart

Install with development dependencies:

```bash
python -m pip install -e '.[dev]'
```

Run test suite and lint checks:

```bash
pytest -q
ruff check src tests
```

Generate runtime catalogues:

```bash
scribekey-models generate
```

Check whether committed generated files are current:

```bash
scribekey-models generate --check
```

Validate canonical metadata, schemas, guardrails, and signatures:

```bash
scribekey-models validate
```

Promote a release to QA or stable channel (fails closed without signing key):

```bash
scribekey-models promote --channel qa --release 2026.09.1 --key-env MODEL_RELEASE_SIGNING_KEY
```

Inspect redistribution-cleared mirror plans (dry-run):

```bash
scribekey-models mirror plan --release 2026.09.1
scribekey-models mirror check
```

## Trust Contract & Promotion Flow

- **Detached Signatures**: Generated channels carry detached P-256 / `SHA256withECDSA` signatures over the exact published bytes.
- **Anti-Rollback Sequence**: Every channel manifest contains a strictly monotonically increasing integer sequence. Emergency rollbacks increment the sequence number while pointing to an older immutable release snapshot.
- **Relative Path Resolution**: Channel manifests point to release snapshots via relative paths (`../releases/<release-id>/<filename>`), preserving deterministic resolution.
- **Reviewable PR Publication Boundary**: Production promotions run via GitHub Actions (`workflow_dispatch`), generating a reviewable PR. Merging the PR into `main` is the atomic publication boundary. Auto-merge is disabled.
- **Failure Safety**: Promotion runs in isolated staging; failures leave the previous stable release intact.

See [docs/operator-guide.md](docs/operator-guide.md) for full operator flow and required CI secrets.
See [docs/compatibility.md](docs/compatibility.md) for consumer runtime compatibility details.
