# Public stream recorder

One-shot dataset collection tooling for Winston. It discovers public HLS webcam streams and records them into hourly MKV files.

YouTube and YouTube Live are intentionally unsupported.

## Requirements

- Python 3.13+
- FFmpeg available on `PATH`
- `pyxlsb`
- Playwright + Chromium for JavaScript-driven players

With `uv`, the one-shot dependencies can stay outside Winston's runtime dependencies:

```bash
uv run --with pyxlsb --with playwright python -m playwright install chromium
```

## Discover one page

```bash
uv run --with pyxlsb --with playwright \
  python -m scripts.stream_recorder.main discover \
  --url 'https://example.com/public-webcam'
```

The command prints JSON containing every candidate `.m3u8` and every resolved logical camera.

## Discover every supported source in Place_Overview.xlsb

```bash
uv run --with pyxlsb --with playwright \
  python -m scripts.stream_recorder.main discover \
  --xlsb Place_Overview.xlsb
```

YouTube rows are skipped automatically.

## Record

```bash
uv run --with pyxlsb --with playwright \
  python -m scripts.stream_recorder.main record \
  --xlsb Place_Overview.xlsb \
  --output datasets/recordings
```

Record only selected source ids:

```bash
uv run --with pyxlsb --with playwright \
  python -m scripts.stream_recorder.main record \
  --xlsb Place_Overview.xlsb \
  --source-id 61 \
  --source-id 82
```

The output layout is:

```text
datasets/recordings/
└── country/city/place/
    ├── source.json
    └── cameras/
        └── camera-001/
            ├── camera.json
            └── YYYY/MM/DD/YYYY-MM-DDTHH-00-00Z.mkv
```

The first file after startup may be shorter than one hour so that subsequent files align with real UTC clock hours.
