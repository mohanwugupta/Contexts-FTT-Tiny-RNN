"""Inner-loop and meta-batch calibration for variable-width Elman SRNs."""

from .config import CalibrationConfig, load_calibration_config


def run_calibration(*args, **kwargs):
    from .pipeline import run_calibration as _run
    return _run(*args, **kwargs)


__all__ = ["CalibrationConfig", "load_calibration_config", "run_calibration"]
