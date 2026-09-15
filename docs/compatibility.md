# Compatibility & Android Consumer Contract

ScribeKey catalogues describe runtime contracts consumed by the Android application.

## Consumer Contract Requirements

The Android client parses published channel manifests (`qa.json`, `stable.json`) and verifies:

1. **Cryptographic Signature**: Detached P-256 / `SHA256withECDSA` signature over exact document bytes.
2. **Anti-Rollback Sequence**: `incoming.sequence > local.sequence`. Updates with non-increasing sequences are rejected.
3. **Compatibility Metadata**:
   ```yaml
   compatibility:
     minAndroidApiLevel: 28
     minAppVersionCode: 10
     catalogsSchemaVersion: 1
     supportedRuntimeFamilies: ["sherpa-onnx", "gguf", "pyannote"]
     supportedConfigFamilies: ["speech-model-catalog", "cleanup-model-catalog", "speaker-diarization-manifest"]
   ```
   - Android will reject catalogues with unsupported runtime or config families.
   - Devices below `minAndroidApiLevel` or app versions below `minAppVersionCode` will decline the update.
4. **Execution Safety**: Android strictly rejects any payload referencing executable extensions (`.apk`, `.dex`, `.so`, `.jar`, `.class`, `.sh`, `.bash`, `.exe`, `.elf`, `.bat`, etc.).
5. **Relative URI Paths**: Catalogs are referenced relative to the channel document:
   `../releases/<release-id>/<filename>`
   The client resolves catalog URLs relative to the channel document location.
