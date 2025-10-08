import importlib
import pytest

@pytest.mark.parametrize(
    "module",
    [
        "concrete_samples_toolkit",
        "concrete_samples_toolkit.data_preprocessing",
        "concrete_samples_toolkit.image_evaluation",
        "concrete_samples_toolkit.image_fusion",
        "concrete_samples_toolkit.utils",
        "concrete_samples_toolkit.video_processing",
        "concrete_samples_toolkit.video_registration",

    ],
)
def test_module_import_test(module):
    importlib.import_module(module)
