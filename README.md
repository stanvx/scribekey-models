# ScribeKey Models

Open model catalogue and distribution tooling for ScribeKey-compatible on-device models.

This repository contains the public, reviewable model metadata that ScribeKey uses to describe supported speech, diarization, and transcript-cleanup models. It also contains the schemas and deterministic generator used to produce the runtime catalogues consumed by the Android app.

The ScribeKey Android application is a separate project. Publishing this repository does not publish the app source code.

## What belongs here

- model identity, compatibility, download and integrity metadata
- immutable upstream revisions and SHA-256 hashes
- licence and attribution metadata
- JSON schemas for catalogue formats
- generated runtime catalogues
- validation and catalogue generation tooling
- release and mirroring automation for artifacts that are explicitly cleared for redistribution

Research datasets, training experiments, private evaluation material, and exploratory model work do not belong in this repository.

## Repository layout

```text
catalog/      Canonical human-edited model metadata
schemas/      JSON schemas for canonical and generated formats
generated/    Deterministic runtime catalogues consumed by ScribeKey
src/          Validation and generation tooling
tests/        Catalogue and reproducibility tests
docs/         Contribution and compatibility notes
```

## Generate the runtime catalogues

```bash
python -m pip install -e '.[dev]'
scribekey-models generate
```

Check whether committed generated files are current:

```bash
scribekey-models generate --check
```

Validate canonical metadata and generated output:

```bash
scribekey-models validate
```

## Model artifacts

Large model files are not committed to Git history. Catalogue entries point at immutable upstream artifacts. ScribeKey may also publish redistribution-cleared mirrors through GitHub Releases or other public mirrors.

Each model keeps its own upstream licence. The repository licence applies to ScribeKey's catalogue metadata and tooling, not to third-party model weights.

## Contributing

See [docs/adding-a-model.md](docs/adding-a-model.md). Pull requests should use immutable upstream revisions, include hashes and licence information, and leave `generated/` reproducible from `catalog/`.

