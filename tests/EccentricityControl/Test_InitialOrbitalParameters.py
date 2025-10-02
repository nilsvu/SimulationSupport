# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy.testing as npt

from SimulationSupport.EccentricityControl.InitialOrbitalParameters import (
    initial_orbital_parameters,
)


def test_initial_orbital_parameters():
    target_params = {
        "MassRatio": 1.0,
        "MassA": 0.5,
        "MassB": 0.5,
        "DimensionlessSpinA": [0.0, 0.0, 0.0],
        "DimensionlessSpinB": [0.0, 0.0, 0.0],
        "Eccentricity": 0.0,
    }
    # Expected results are computed from SpEC's ZeroEccParamsFromPN.py
    npt.assert_allclose(
        initial_orbital_parameters(
            target_params,
            separation=20.0,
            orbital_angular_velocity=0.01,
            radial_expansion_velocity=-1.0e-5,
        ),
        [20.0, 0.01, -1.0e-5],
    )
    npt.assert_allclose(
        initial_orbital_parameters(
            target_params,
            separation=16.0,
        ),
        [16.0, 0.014474280975952748, -4.117670632867514e-05],
    )
    npt.assert_allclose(
        initial_orbital_parameters(
            target_params,
            orbital_angular_velocity=0.015,
        ),
        [15.6060791015625, 0.015, -4.541705362753467e-05],
    )
    npt.assert_allclose(
        initial_orbital_parameters(
            target_params,
            orbital_angular_velocity=0.015,
        ),
        [15.6060791015625, 0.015, -4.541705362753467e-05],
    )
    npt.assert_allclose(
        initial_orbital_parameters(
            {**target_params, "NumOrbits": 20},
        ),
        [16.0421142578125, 0.014419921875000002, -4.0753460821644916e-05],
    )
    npt.assert_allclose(
        initial_orbital_parameters(
            {**target_params, "TimeToMerger": 6000},
        ),
        [16.1357421875, 0.01430025219917298, -3.9831982447244026e-05],
    )
