from pathlib import Path

import pytest

from winston.config import Settings


def test_settings_reads_unprefixed_environment_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FFPROBE_BINARY", "custom-ffprobe")

    configured = Settings(_env_file=None)

    assert configured.data_dir == tmp_path
    assert configured.ffprobe_binary == "custom-ffprobe"
