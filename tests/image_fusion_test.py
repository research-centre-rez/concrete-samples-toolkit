import pytest
import cv2 as cv
import numpy as np
from concrete_registration_toolkit.image_fusion import __main__
import concrete_registration_toolkit.image_fusion as mut
from .utils import dummy_video_capture_image_proxy


@pytest.mark.parametrize(
    "method",
    [
        ["min"],
        ["var", "max"],
        ["med", "var", "max"],
        ["mean","med", "var", "max"],
        ["all"],
    ],
)
def test_parse_args(monkeypatch, method):
    dummy_args = [
        "prog",
        "-i",
        "input_path",
        "-o",
        "output_path",
        "-m",
        *method,
        "--crop",
    ]

    monkeypatch.setattr("sys.argv", dummy_args)

    args = __main__.parse_args()

    assert args.input == ["input_path"]
    assert args.output == "output_path"
    assert args.method == method
    assert args.crop

@pytest.mark.parametrize(
    "args_list",
    [
        ["prog"],
        ["prog", "-i", "input_path"],
        ["prog", "-i", "input_path", "-o", "output_path", "-m", "invalid_method"],
    ]
)
def test_parse_args_invalid(monkeypatch, args_list, capsys):
    monkeypatch.setattr("sys.argv", args_list)

    with pytest.raises(SystemExit) as e:
        __main__.parse_args()

    assert e.value.code != 0

    captured = capsys.readouterr()
    assert "error:" in captured.err

@pytest.mark.parametrize(
    "fuse_method, expected_class",
    [
        (mut.FuseMethod.MIN, "MinFuser"),
        (mut.FuseMethod.MAX, "MaxFuser"),
        (mut.FuseMethod.VAR, "VarFuser"),
        (mut.FuseMethod.MED, "MedianFuser"),
        (mut.FuseMethod.MEAN, "MeanFuser"),
    ]
)
def test_fuse_factory(fuse_method, expected_class):
    factory = mut.ImageFuserFactory()

    fuser = factory.get_fuser(fuse_method)
    assert isinstance(fuser, getattr(mut.factory_fuser, expected_class))


@pytest.mark.parametrize(
    "fuse_method",
    [
        mut.FuseMethod.MIN,
        mut.FuseMethod.MAX,
        mut.FuseMethod.VAR,
        mut.FuseMethod.MED,
        mut.FuseMethod.MEAN,
    ]
)
def test_image_fusion(fuse_method, dummy_video_capture_image_proxy, monkeypatch):

    def dummy_verify_video_stack(self, video_stack):
        stack = []
        for _ in range(dummy_video_capture_image_proxy.get(cv.CAP_PROP_FRAME_COUNT)):
            ret, image = dummy_video_capture_image_proxy.read()
            if not ret:
                break
            stack.append(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
        return np.array(stack)

    monkeypatch.setattr(mut.factory_fuser.Fuser, '_verify_video_stack', dummy_verify_video_stack)


    factory = mut.ImageFuserFactory()

    fuser = factory.get_fuser(fuse_method)
    expected_image_shape = (dummy_video_capture_image_proxy.get(cv.CAP_PROP_FRAME_HEIGHT), dummy_video_capture_image_proxy.get(cv.CAP_PROP_FRAME_WIDTH))

    image = fuser.get_fused_image('dummy_path')
    assert isinstance(image, np.ndarray)
    assert image.shape == expected_image_shape

@pytest.mark.parametrize(
    "fuse_method",
    [
        mut.FuseMethod.MIN,
        mut.FuseMethod.MAX,
        mut.FuseMethod.VAR,
        mut.FuseMethod.MED,
        mut.FuseMethod.MEAN,
    ]
)
def test_image_writing(fuse_method, tmp_path):
    factory = mut.ImageFuserFactory()

    fuser = factory.get_fuser(fuse_method)

    # Create dummy image
    fused_image = np.full((200, 200, 3), 128, dtype=np.uint8)

    # Expected save path (tmp dir)
    save_as = str(tmp_path / "test_output.jpg")  # wrong extension on purpose

    # Monkeypatch cv.imwrite so we don’t actually write a file if you want
    # But here, let's actually check file creation
    fuser.save_image_to_disc(fused_image, save_as)

    # Because invalid extension ".jpg" should be converted to ".png"
    expected_file = tmp_path / f"test_output_{fuser.method_name}.png"
    assert expected_file.exists()

    # Check image was saved correctly (shape + pixel values)
    saved_img = cv.imread(str(expected_file))
    assert saved_img is not None
    assert saved_img.shape == fused_image.shape
    # OpenCV might load as BGR but values should match 128
    assert np.allclose(saved_img, 128, atol=1)
