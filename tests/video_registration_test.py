import pytest

import os
import numpy as np
import cv2 as cv
import pytest
import jsonschema

import concrete_registration_toolkit.video_registration as vr
from concrete_registration_toolkit.video_registration import __main__
from concrete_registration_toolkit.utils import load_config, load_json_schema

from .utils import dummy_video_capture, dummy_video_capture_image_proxy


@pytest.fixture
def sample_config():
    root = os.path.dirname(vr.__file__)
    config = load_config(os.path.join(root, "default_config.json5"))
    config["mudic"]["mesh_parameters"] = {
        "box_h": 5,
        "box_w": 5,
        "num_elems_x": 5,
        "num_elems_y": 5,
    }
    return config


@pytest.fixture
def registrator_factory(dummy_video_capture_image_proxy, monkeypatch):

    def dummy_create_video_matrix(input_video, grayscale, max_gb_memory=1):
        stack = []
        for _ in range(dummy_video_capture_image_proxy.get(cv.CAP_PROP_FRAME_COUNT)):
            ret, image = dummy_video_capture_image_proxy.read()
            if not ret:
                break
            stack.append(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
        return np.array(stack)

    def _factory(method: vr.RegMethod, config):
        monkeypatch.setattr(
            vr.registrator,
            "prep_cap",
            lambda path, set_to: dummy_video_capture_image_proxy,
        )
        monkeypatch.setattr(
            vr.registrator, "create_video_matrix", dummy_create_video_matrix
        )
        return vr.VideoRegistrator(method, config)

    return _factory


@pytest.fixture
def mudic_registrator(sample_config, registrator_factory):
    return registrator_factory(vr.RegMethod.MUDIC, sample_config)


@pytest.fixture
def orb_registrator(sample_config, registrator_factory):
    return registrator_factory(vr.RegMethod.ORB, sample_config)


@pytest.fixture
def lightglue_registrator(sample_config, registrator_factory):
    return registrator_factory(vr.RegMethod.LIGHTGLUE, sample_config)


@pytest.mark.parametrize(
    "method",
    [
        "mudic",
        "orb",
        "lightglue",
    ],
)
def test_parse_args(monkeypatch, method):
    dummy_args = [
        "prog",
        "-i",
        "input_path",
        "-o",
        "output_path",
        "-c",
        "config_path",
        "-m",
        method,
    ]

    monkeypatch.setattr("sys.argv", dummy_args)

    args = __main__.parse_args()

    assert args.input == ["input_path"]
    assert args.output == "output_path"
    assert args.config == "config_path"
    assert args.method == method


@pytest.mark.parametrize(
    "args_list",
    [
        ["prog"],
        ["prog", "-i", "input_path"],
        ["prog", "-i", "input_path", "-o", "output_path", "-m", "invalid_method"],
    ],
)
def test_parse_args_invalid(monkeypatch, args_list, capsys):
    monkeypatch.setattr("sys.argv", args_list)

    with pytest.raises(SystemExit) as e:
        __main__.parse_args()

    assert e.value.code != 0

    captured = capsys.readouterr()
    assert "error:" in captured.err


def test_verify_config():
    root = os.path.dirname(vr.__file__)
    config = load_config(os.path.join(root, "default_config.json5"))
    schema = load_json_schema(os.path.join(root, "video_registration_schema.json"))

    assert isinstance(config, dict)
    assert isinstance(schema, dict)

    jsonschema.validate(instance=config, schema=schema)

    invalid_config = {"invalid_key": "invalid_value", "another_invalid_key": 0}
    with pytest.raises(jsonschema.ValidationError) as _:
        jsonschema.validate(instance=invalid_config, schema=schema)


@pytest.mark.parametrize(
    "method, expected_method",
    [
        (vr.RegMethod.MUDIC, "_get_mudic_registration"),
        (vr.RegMethod.ORB, "_get_orb_registration"),
        (vr.RegMethod.LIGHTGLUE, "_get_lightglue_registration"),
    ],
)
def test_class_instance(method, expected_method, sample_config):
    registrator = vr.VideoRegistrator(method=method, config=sample_config)
    assert registrator.method == getattr(registrator, expected_method)


@pytest.mark.parametrize(
    "registrator_fixture",
    [
        "mudic_registrator",
        "orb_registrator",
        "lightglue_registrator",
    ],
)
def test_get_registered_block(
    registrator_fixture, request, monkeypatch, dummy_video_capture_image_proxy
):
    video = dummy_video_capture_image_proxy

    registrator = request.getfixturevalue(registrator_fixture)
    results = registrator.get_registered_block("dummy_path")
    reg_block = results["registered_block"]
    transformations = results["transformations"]

    vid_frames = video.get(cv.CAP_PROP_FRAME_COUNT)
    vid_height = video.get(cv.CAP_PROP_FRAME_HEIGHT)
    vid_width = video.get(cv.CAP_PROP_FRAME_WIDTH)
    expected_stack_shape = (vid_frames, vid_height, vid_width)
    expected_transformation_shape = (5, 3, 3)

    assert isinstance(results, dict)
    assert isinstance(reg_block, np.ndarray)
    assert isinstance(transformations, np.ndarray)
    assert transformations.shape == expected_transformation_shape
    assert reg_block.shape == expected_stack_shape


@pytest.mark.parametrize(
    "registrator_fixture",
    [
        "mudic_registrator",
        "orb_registrator",
        "lightglue_registrator",
    ],
)
def test_write_out_npy_matrix(registrator_fixture, request, monkeypatch, tmp_path):
    registrator = request.getfixturevalue(registrator_fixture)
    called = {}

    def dummy_write_trans_into_csv(trans, save_as):
        called["save_as"] = save_as
        for t in trans:
            assert t.shape == (3, 3)

    def dummy_np_save(block, name):
        called["np_save"] = True

    monkeypatch.setattr(
        registrator, "_write_transformation_into_csv", dummy_write_trans_into_csv
    )
    monkeypatch.setattr(vr.registrator.np, "save", dummy_np_save)

    results = registrator.get_registered_block("dummy_path")

    save_as = str(tmp_path / "output.npy")

    registrator.save_registered_block(results, save_as)

    assert called["save_as"] == save_as
    assert "np_save" in called
