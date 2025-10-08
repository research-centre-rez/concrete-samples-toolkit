import pytest
from concrete_samples_toolkit.image_evaluation import __main__
import concrete_samples_toolkit.image_evaluation as mut

import numpy as np


@pytest.mark.parametrize(
    "norm_type",
    ["l1_norm", "grad_mag"],
)
def test_parse_args(monkeypatch, norm_type):
    dummy_args = [
        "prog",
        "-i",
        "input_path",
        "-o",
        "output_path",
        "-n",
        norm_type,
    ]

    monkeypatch.setattr("sys.argv", dummy_args)

    args = __main__.parse_args()

    assert args.input == ["input_path"]
    assert args.output == "output_path"
    assert args.normalisation_type == norm_type


@pytest.mark.parametrize(
    "args_list",
    [
        ["prog"],
        ["prog", "-i", "input_path"],
        ["prog", "-i", "input_path", "-o", "output_path", "-n", "invalid_method"],
    ],
)
def test_parse_args_invalid(monkeypatch, args_list, capsys):
    monkeypatch.setattr("sys.argv", args_list)

    with pytest.raises(SystemExit) as e:
        __main__.parse_args()

    assert e.value.code != 0

    captured = capsys.readouterr()
    assert "error:" in captured.err




@pytest.fixture
def dummy_image():
    img = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1))
    return img


@pytest.fixture
def dummy_stack(dummy_image):
    stack = np.stack(
        [dummy_image, np.roll(dummy_image, 1, axis=0), np.roll(dummy_image, 2, axis=1)],
        axis=0,
    )
    return stack.astype(np.uint8)


@pytest.mark.parametrize(
    "norm_type", [mut.metrics.normType.l1_norm, mut.metrics.normType.grad_mag]
)
def test_normalise_img(dummy_image, norm_type):
    metric = mut.metrics.Metric()
    result = metric._normalise_img(dummy_image, norm_type)
    assert isinstance(result, np.ndarray)
    assert result.shape == dummy_image.shape
    assert np.all(np.isfinite(result))


def test_mask_sample(dummy_image):
    metric = mut.metrics.Metric()
    masked = metric._mask_sample(dummy_image)
    assert masked.shape == dummy_image.shape
    # mask should zero out corners
    assert np.any(masked == 0)


@pytest.mark.parametrize(
    "normalise,norm_type", [(False, None), (True, mut.metrics.normType.l1_norm)]
)
def test_preprocess_image_array(dummy_image, normalise, norm_type):
    metric = mut.metrics.Metric()
    out = metric._preprocess_image(dummy_image, normalise, norm_type)
    assert isinstance(out, np.ndarray)
    assert out.min() >= 0.0


def test_nglv_metric(dummy_image):
    nglv = mut.NGLV()
    score = nglv.calculate_metric(dummy_image, True, mut.metrics.normType.l1_norm)
    assert isinstance(score, float)
    assert score >= 0


def test_brenner_metric(dummy_image):
    brenner = mut.BrennerMethod()
    score = brenner.calculate_metric(dummy_image, True, mut.metrics.normType.l1_norm)
    assert isinstance(score, float)
    assert score >= 0


def test_absolute_gradient(dummy_image):
    grad = mut.AbsoluteGradient()
    score = grad.calculate_metric(dummy_image, True, mut.metrics.normType.l1_norm)
    assert isinstance(score, float)
    assert score >= 0


def test_mutual_information_metric(dummy_stack):
    mi = mut.MutualInformation()
    score = mi.calculate_metric(dummy_stack, normalise=True)
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_mutual_information_with_std(dummy_stack):
    mi = mut.MutualInformation()
    mean_score, std_score = mi.calculate_metric_with_std(dummy_stack, normalise=True)
    assert isinstance(mean_score, float)
    assert isinstance(std_score, float)
    assert mean_score >= 0
    assert std_score >= 0
