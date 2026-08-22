from pathlib import Path

import pytest

from winston.config import Settings, get_config


def test_get_config_reads_unprefixed_environment_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FFMPEG_BINARY", "custom-ffmpeg")
    monkeypatch.setenv("FFPROBE_BINARY", "custom-ffprobe")

    configured = get_config()

    assert configured.data_dir == tmp_path
    assert configured.ffmpeg_binary == "custom-ffmpeg"
    assert configured.ffprobe_binary == "custom-ffprobe"


def test_get_config_returns_a_new_settings_instance() -> None:
    assert get_config() is not get_config()
