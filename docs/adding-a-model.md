# Adding a model

The canonical model definitions live in `catalog/`. Generated JSON in `generated/` should not be edited by hand.

For a new model:

1. Add or update the relevant YAML file in `catalog/`.
2. Pin the upstream artifact to an immutable revision where the provider supports it.
3. Record the expected file size and SHA-256 digest.
4. Include upstream licence and attribution information.
5. Run `scribekey-models generate`.
6. Run `scribekey-models validate` and `pytest`.

Model weights are not added to Git history. Mirroring is handled separately and only for artifacts whose licence permits redistribution.

Compatibility metadata should describe what ScribeKey actually supports rather than what an upstream runtime might support in theory.

