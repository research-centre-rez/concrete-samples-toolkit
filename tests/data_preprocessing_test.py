import pytest
from types import SimpleNamespace

# Import __main__ module
from concrete_registration_toolkit.data_preprocessing import __main__ as main_module
import concrete_registration_toolkit.data_preprocessing.detect_black_frames as dbf
import concrete_registration_toolkit.data_preprocessing.split_video as mut


@pytest.fixture
def mock_args(tmp_path):
    """Create dummy args object with temporary directories."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    return SimpleNamespace(input=str(input_dir), output=str(output_dir))


def test_parse_args(monkeypatch):
    """Test parse_args parses CLI arguments correctly."""
    test_args = ["aa", "-i", "input_path", "-o", "output_path"]
    monkeypatch.setattr("sys.argv", test_args)
    args = main_module.parse_args()

    assert args.input == "input_path"
    assert args.output == "output_path"


def test_split_videos_in_empty_directory(tmp_path, monkeypatch):
    """_split_videos_in_directory does nothing if it is called on empty dir."""

    output_dir = tmp_path / "output"
    called = {"split_video": False}

    def fake_split_video(vid_path, out_path):
        called["split_video"] = True

    monkeypatch.setattr(main_module, "_split_video", fake_split_video)
    main_module._split_videos_in_directory(str(tmp_path), str(output_dir))

    assert called["split_video"] is False
    assert not output_dir.exists()


def test_split_video_sucess(tmp_path, monkeypatch):
    """_split_video calls detect_black_frames and split_video"""

    vid_path = tmp_path / "video.mp4"
    vid_path.touch()
    out_path = tmp_path / "output"

    def fake_detect_black_frames(path):
        assert path == str(vid_path)
        return [], 30

    def fake_split_video(path, black_frames, out_dir, fps):
        assert path == str(vid_path)
        assert black_frames == []
        assert out_dir == out_path
        assert fps == 30

        return {"part1": 10, "part2": 20}

    monkeypatch.setattr(main_module, "detect_black_frames", fake_detect_black_frames)
    monkeypatch.setattr(main_module, "split_video", fake_split_video)

    main_module._split_video(str(vid_path), str(out_path))

    assert out_path.exists()


def test_main_directory_vs_file(tmp_path, monkeypatch, mock_args):
    """Test main() chooses _split_videos_in_directory or _split_video correctly."""
    called = {"dir": False, "file": False}

    def fake_split_videos_in_directory(directory_path, output_dir):
        called["dir"] = True
        assert directory_path == mock_args.input
        assert output_dir == mock_args.output

    def fake_split_video(vid_path, out_path):
        called["file"] = True
        assert vid_path == mock_args.input
        assert out_path == mock_args.output

    monkeypatch.setattr(
        main_module, "_split_videos_in_directory", fake_split_videos_in_directory
    )
    monkeypatch.setattr(main_module, "_split_video", fake_split_video)

    # Directory case
    main_module.main(mock_args)
    assert called["dir"] is True
    assert called["file"] is False

    # Single file case
    called["dir"] = False
    called["file"] = False
    file_path = tmp_path / "video.mp4"
    file_path.touch()
    # Update args.input to be a file
    mock_args.input = str(file_path)
    main_module.main(mock_args)
    assert called["file"] is True
    assert called["dir"] is False


def test_detect_black_frames(monkeypatch):
    class DummyCapture:
        def __init__(self, path):
            self.frames = [
                (True, "frame1"),
                (True, "frame2"),
                (True, "frame3"),
                (True, "frame4"),
                (True, "frame5"),
                (True, "frame6"),
                (False, "null_frame"),  # End of video frame
            ]
            self.index = 0

        def isOpened(self):
            return True

        def get(self, property):
            # cv.CAP_PROP_FPS
            if property == 5:
                return 30.0

            # cv.CAP_PROP_FRAME_COUNT
            elif property == 7:
                return len(self.frames)

        def read(self):
            frame = self.frames[self.index]
            self.index += 1
            return frame

        def release(self):
            pass

    monkeypatch.setattr(dbf.cv, "VideoCapture", DummyCapture)

    values = [0, 10, 12, 11, 0, 0]

    def dummy_grid_sample_mean(frame, n):
        return values.pop(0)

    monkeypatch.setattr(dbf, "grid_sample_mean", dummy_grid_sample_mean)

    black_indices, fps = dbf.detect_black_frames("dummy_path.mp4", threshold=5)

    assert fps == 30.0
    assert black_indices == [1, 3]


def test_split_video(monkeypatch, tmp_path):

    # We don't actually want to call ffmpeg
    def fake_ffmpeg_call(cmd):
        return 0

    monkeypatch.setattr(mut.subprocess, "call", fake_ffmpeg_call)

    video_path = "input-video.mp4"
    frame_idx = [0, 30, 60, 90]
    fps = 30.0

    stats = mut.split_video(video_path, frame_idx, str(tmp_path), fps)

    # Check whether the directory was successfully built
    expected_dir = tmp_path / "input"
    assert expected_dir.exists()

    # Checking stat validity, i.e. they have mp4 videos and each video has length of 1 second
    assert len(stats.keys()) == 2
    for key, value in stats.items():
        assert key.endswith(".mp4")
        assert value == pytest.approx(1.0)
