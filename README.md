# EYEAR Node

Room-level sensing and local perception for [EYEAR](https://github.com/HY-LG-EYEAR).

**Maintainer:** Banseok Shin

The Node processes camera, microphone, and mmWave observations and sends semantic events to the World Model. It also plays speech requests from the Orchestrator and returns transcribed replies or direct commands. Raw camera and microphone streams remain local.

## Code

| Directory | Purpose |
| --- | --- |
| `vision/` | Camera, face identification, posture, and brightness |
| `audio/` | Speech detection, transcription, playback, and voice sessions |
| `radar/` | Radar module |
| `fusion/` | Local observation fusion and event construction |
| `publisher/` | Event publishing and simulated input |
| `recorder/` | Recording and replay module |
| `schema/` | Message models, JSON Schemas, and examples |
| `config/` | Node configuration |
| `tests/` | Tests |

Python 3.11. Dependencies are listed in [pyproject.toml](pyproject.toml).

## Interface

[Node message specification](docs/schema_v0.md) · [Message examples](schema/examples/)
