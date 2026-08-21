# Winston

Find anything in video footage with natural-language search and multimodal embeddings. No predefined classes or tracking required.

Winston is being built as an offline-first semantic search engine for recorded surveillance footage and photos. The baseline design uses sparse video keyframe sampling, class-agnostic image regions, Jina CLIP v1 multimodal embeddings, and Qdrant for vector retrieval.

## Design

- [Architecture](docs/architecture.md)
- [Roadmap](docs/roadmap.md)

## Current workstreams

- `feat/media-ingest` — local media discovery/probing; draft PR #1 is open against `main`.
- `feat/public-stream-recorder` — one-shot public HLS dataset collection tooling under `scripts/`; intentionally separate from Winston runtime code.

The first algorithmic milestone is natural-language retrieval of visual concepts that were never declared as detector classes during indexing, such as `woman with a pink stroller`.
