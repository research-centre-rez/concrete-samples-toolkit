import os
import numpy as np
import cv2 as cv
import pytest
import jsonschema

from .utils import dummy_video_capture

import concrete_registration_toolkit.video_processing as mut
from concrete_registration_toolkit.video_processing import __main__
from concrete_registration_toolkit.utils import load_config, load_json_schema



@pytest.fixture
def sample_config():
    root = os.path.dirname(mut.__file__)
    config = load_config(os.path.join(root, "default_config.json5"))
    config["start_at"] = 0
    config["rough_rotation_estimation"]["rotation_center"]["x"] = 100
    config["rough_rotation_estimation"]["rotation_center"]["y"] = 100
    config["rough_rotation_estimation"]["rotation_per_frame"] = 10

    return config

@pytest.fixture
def processor_factory(dummy_video_capture, monkeypatch):
    def _factory(method: mut.ProcessorMethod, config):
        monkeypatch.setattr(mut.video_processor, "prep_cap", lambda path, start: dummy_video_capture)
        return mut.VideoProcessor(method, config)
    return _factory

@pytest.fixture
def none_processor(sample_config, processor_factory):
    return processor_factory(mut.ProcessorMethod.NONE, sample_config)


@pytest.fixture
def approx_processor(sample_config, processor_factory):
    return processor_factory(mut.ProcessorMethod.APPROX, sample_config)


@pytest.fixture
def optical_flow_processor(sample_config, processor_factory, monkeypatch, dummy_video_capture):
    def dummy_analyse_sparse_optical_flow(video_path, lk_params, f_params, start_at):
        trajectories = []
        for i in range(dummy_video_capture.get(cv.CAP_PROP_FRAME_COUNT)):
            trajectories.append(np.array([i, i]))

        return trajectories

    def dummy_estimate_rotation_center_for_each_trajectory(trajectories, mode):
        center = np.array([100, 100])
        error = np.array(1.0)
        return (center, error)

    def dummy_calculate_angular_movement(np_trajectories, center):
        angles = [0]
        for _ in range(len(np_trajectories) - 1):
            angles.append(10)
        results = {"median_angle_per_frame_deg": angles}
        return results

    def dummy_visualise_rotation_analysis(np_trajectories, rotation_res, graph_config):
        return None

    monkeypatch.setattr(
        mut.video_processor,
        "analyse_sparse_optical_flow",
        dummy_analyse_sparse_optical_flow,
    )
    monkeypatch.setattr(
        mut.video_processor,
        "estimate_rotation_center_for_each_trajectory",
        dummy_estimate_rotation_center_for_each_trajectory,
    )
    monkeypatch.setattr(
        mut.video_processor,
        "calculate_angular_movement",
        dummy_calculate_angular_movement,
    )
    monkeypatch.setattr(
        mut.video_processor.visualisers,
        "visualize_rotation_analysis",
        dummy_visualise_rotation_analysis,
    )
    return processor_factory(mut.ProcessorMethod.OPT_FLOW, sample_config)



def test_parse_args(monkeypatch):
    dummy_args = [
        "prog",
        "-i",
        "input_path",
        "-o",
        "output_path",
        "-c",
        "config_path",
        "-m",
        "opt_flow",
    ]

    monkeypatch.setattr("sys.argv", dummy_args)

    args = __main__.parse_args()

    assert args.input == ["input_path"]
    assert args.output == "output_path"
    assert args.config == "config_path"
    assert args.method == "opt_flow"

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


def test_verify_config():
    root = os.path.dirname(mut.__file__)

    config = load_config(os.path.join(root, "default_config.json5"))
    schema = load_json_schema(os.path.join(root, "video_processing_schema.json"))

    assert isinstance(config, dict)
    assert isinstance(schema, dict)

    jsonschema.validate(instance=config, schema=schema)

    invalid_config = {
        "invalid_key": "invalid_value",
        "another_invalid_key": 0
    }
    with pytest.raises(jsonschema.ValidationError) as e:
        jsonschema.validate(instance=invalid_config, schema=schema)


@pytest.mark.parametrize(
    "method, expected_method", 
    [
        (mut.ProcessorMethod.NONE, "_get_none_analysis"),
        (mut.ProcessorMethod.APPROX, "_get_estimate_analysis"),
        (mut.ProcessorMethod.OPT_FLOW, "_get_optical_flow_analysis"),
    ]
)
def test_class_instance(method, expected_method, sample_config):
    processor = mut.VideoProcessor(method=method, config=sample_config)
    assert processor.method == getattr(processor, expected_method)



@pytest.mark.parametrize(
    "processor_fixture, expected_center, expected_sum",
    [
        ("none_processor", [0, 0], 0),
        ("approx_processor", [100, 100], 40),
        ("optical_flow_processor", [100, 100], 40),
    ],
)
def test_get_rotation_analysis(
    processor_fixture, expected_center, expected_sum, request
):
    processor = request.getfixturevalue(processor_fixture)
    analysis = processor.get_rotation_analysis("dummy_path")

    assert isinstance(analysis, dict)
    assert isinstance(analysis["center"], (list, np.ndarray))
    assert len(analysis["angles"]) == 5
    assert sum(analysis["angles"]) == expected_sum
    assert list(analysis["center"]) == expected_center


@pytest.mark.parametrize(
    "processor_fixture, center",
    [
        ("none_processor", [0.0, 0.0]),
        ("approx_processor", [100.0, 100.0]),
        ("optical_flow_processor", [100.0, 100.0]),
    ],
)
def test_write_out_video(
    processor_fixture, center, request, dummy_video_capture, monkeypatch, tmp_path
):
    processor = request.getfixturevalue(processor_fixture)
    called = {}

    def dummy_write_trans_into_csv(trans, save_as):
        called["save_as"] = save_as
        for t in trans:
            assert t.shape == (2, 3)

    monkeypatch.setattr(processor, "_write_trans_into_csv", dummy_write_trans_into_csv)

    # Create dummy analysis for 5 frames
    analysis = processor.get_rotation_analysis("dummy_path")
    analysis["center"] = np.array(center)

    save_as = str(tmp_path / "output.mp4")
    processor.write_out_video("dummy_path", analysis, save_as)

    assert os.path.exists(save_as)
    assert called["save_as"] == save_as
