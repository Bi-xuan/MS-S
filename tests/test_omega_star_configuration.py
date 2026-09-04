"""Tests for the project-wide synthetic omega_star configuration."""

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import admm
from experiments import compute_objective_curve, run_synthetic


@pytest.mark.parametrize(
    "parse_args",
    [
        admm.parse_args,
        compute_objective_curve.parse_args,
        run_synthetic.parse_args,
    ],
)
def test_omega_star_defaults_to_project_default(parse_args):
    args = parse_args([])

    assert args.omega_star == pytest.approx(admm.DEFAULT_OMEGA_STAR)
    assert args.omega_star == pytest.approx(0.1)


@pytest.mark.parametrize(
    "parse_args",
    [
        admm.parse_args,
        compute_objective_curve.parse_args,
        run_synthetic.parse_args,
    ],
)
def test_omega_star_can_be_overridden(parse_args):
    args = parse_args(["--omega-star", "0.35"])

    assert args.omega_star == pytest.approx(0.35)
