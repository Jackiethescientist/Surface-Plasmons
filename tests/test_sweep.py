import pytest

from surface_plasmons.sweep import (
    SweepConfig,
    expand_sweep,
    linear_axis,
    set_config_value,
)


def test_grid_sweep_expansion_and_repetitions() -> None:
    sweep = SweepConfig.from_dict(
        {
            "base_config": {
                "geometry_name": "nanoparticle_sphere",
                "geometry_parameters": {"particle": "au"},
                "resolution": 60,
            },
            "parameters": {
                "geometry_parameters.radius": [0.03, 0.04],
                "resolution": [60, 80],
            },
            "repetitions": 2,
        }
    )

    expanded = expand_sweep(sweep)

    assert len(expanded) == 8
    assert {candidate["resolution"] for candidate, _, _ in expanded} == {60, 80}
    assert {
        candidate["geometry_parameters"]["radius"]
        for candidate, _, _ in expanded
    } == {0.03, 0.04}


def test_nested_list_path_and_linear_axis() -> None:
    config = {
        "geometry_parameters": {
            "layers": [
                {"thickness": 0.02},
                {"thickness": 0.03},
            ]
        }
    }
    set_config_value(config, "geometry_parameters.layers[1].thickness", 0.05)

    assert config["geometry_parameters"]["layers"][1]["thickness"] == 0.05

    axis = linear_axis("resolution", 20, 80, 4)
    assert axis.values == pytest.approx((20.0, 40.0, 60.0, 80.0))
