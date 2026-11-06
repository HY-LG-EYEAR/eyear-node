# EYEAR Node

Owner: **Banseok Shin** | Part of [EYEAR](https://github.com/HY-LG-EYEAR)

> Status: repository scaffold and implementation plan. This upload kit does not contain runnable services, approved new contracts, trained models, or measured results. Preserve existing implementation files when adding this scaffold.

## Responsibility

Local sensor acquisition, specialized pretrained perception, modality fusion, and semantic event publication. Camera, microphone, speaker, and compute use a macOS laptop surrogate; HLK-LD2450 and LD2410C cover tracking and static presence.

Planned components: InsightFace-based identification, MediaPipe pose landmarks with project-defined posture rules, MobileCLIP zero-shot matching, YAMNet audio context mapping, local speech recognition, and local speech output. Exact versions, thresholds, runtime performance, and the speech/wake-phrase implementation require measurement.

RESTING is a project inference, not proof of sleep. Preserve uncertainty under occlusion and darkness. Track-carried identity must have explicit continuity and expiry conditions. The Node transcribes an answer; the central service interprets it.

## Inputs and outputs

Semantic events go to the World Model. Speech requests arrive from the central service, and correlated replies/direct commands return through the agreed route. Raw camera/audio stays inside the local perception boundary.

The authoritative shared contract release is maintained in [LG-EYEAR/contracts](https://github.com/HY-LG-EYEAR/LG-EYEAR/tree/main/contracts). Coordinate changes with producers and consumers; do not create an incompatible local contract under the same version.

## Structure

| Path | Intended content |
| --- | --- |
| `src/vision/` | Face enrollment/matching, pose and rest evidence; no home-wide policies. |
| `src/audio/` | Audio contexts, speech gating, local transcription and playback. Do not transcribe silence as an answer. |
| `src/radar/` | Vendor-protocol parsing, availability, presence and tracks, with mocks. |
| `src/fusion/` | Time alignment, association, confidence, explicit UNKNOWN and occupied-without-track states. |
| `src/publisher/` | Versioned semantic events and heartbeat/state-change publication. |
| `src/dialogue/` | Execute central speech requests and correlate replies; do not decide appliance actions. |
| `configs/` | Document device ports, calibration and model configuration; keep personal enrollment material local. |
| `replay/` | Record/replay tooling with source timestamps and explicit consent boundaries. |
| `tests/` | Unit/adapter tests using small non-sensitive fixtures and simulated sensor failures. |

## Setup and execution status

This scaffold introduces no runtime dependencies or working launch commands. Preserve existing code. The owner must add the actual prerequisites, pinned dependency installation, configuration example, start/stop commands, and one verified smoke-test procedure before marking this repository runnable. Separate existing prototype instructions from the planned integrated system.

## Verification checklist

- [ ] Shared contract version and valid/invalid examples are identified.
- [ ] Dependencies and OS/hardware versions are recorded.
- [ ] A mock-input smoke test is reproducible.
- [ ] Missing, stale, unknown and invalid input behavior is tested.
- [ ] Integration with the next component is verified and exact commits are recorded.
- [ ] Current limitations and measured results are documented without invented metrics.

## Documentation

- [Integration and scenarios](https://github.com/HY-LG-EYEAR/LG-EYEAR)
- [CSE406 engineering document](https://github.com/HY-LG-EYEAR/LG-EYEAR-docs)
- [ITE351 research blog](https://github.com/HY-LG-EYEAR/LG-EYEAR-blog)

> Independent academic prototype. EYEAR is not an official LG Electronics product. The course prototype does not modify ThinQ ON firmware, integrate with the production ThinQ application, or control physical LG appliances.
