# Compatibility

ScribeKey catalogues describe runtime contracts used by the Android application.

Speech models currently describe Sherpa-ONNX layouts. Cleanup models describe local GGUF models used by the transcript cleanup runtime. Diarization metadata describes the segmentation and embedding assets used by the speaker pipeline.

An entry being present in this repository means ScribeKey knows how to consume that exact artifact shape. It does not imply that every artifact from the same upstream model family is compatible.

The catalogue intentionally records immutable revisions, checksums, Android API requirements, memory guidance, and runtime-specific metadata so the app can make conservative compatibility decisions.

