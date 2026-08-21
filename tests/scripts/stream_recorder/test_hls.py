from scripts.stream_recorder.hls import parse_master_playlist, choose_best_stream


MASTER = '''#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=640x360
low/index.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=2400000,RESOLUTION=1280x720
https://cdn.example/high/index.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=5000000,RESOLUTION=1920x1080
full/index.m3u8
'''


def test_parse_master_playlist_resolves_variants():
    variants = parse_master_playlist(MASTER, "https://origin.example/live/master.m3u8")
    assert [v.resolution for v in variants] == [(640, 360), (1280, 720), (1920, 1080)]
    assert variants[-1].url == "https://origin.example/live/full/index.m3u8"


def test_choose_best_stream_prefers_highest_resolution_then_bandwidth():
    variants = parse_master_playlist(MASTER, "https://origin.example/live/master.m3u8")
    assert choose_best_stream(variants).resolution == (1920, 1080)
