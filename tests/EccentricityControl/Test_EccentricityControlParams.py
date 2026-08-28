# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

from SimulationSupport.EccentricityControl.EccentricityControlParams import (
    eccentricity_control_params,
)


def binary_trajectories(times, initial_separation):
    """Leading-order post-Newtonian trajectories of an equal-mass binary.

    Implements Eqs. (226) and (228) of Blanchet (2013) at leading order, same
    as the 'BinaryTrajectories' test helper in SpECTRE.
    """
    separation = (initial_separation**4 - 64.0 / 5.0 * times) ** 0.25
    orbital_frequency = separation**-1.5
    angle = orbital_frequency * times
    offset = 0.5 * separation * np.array([np.cos(angle), np.sin(angle)])
    zeros = np.zeros_like(times)
    trajectory_a = np.column_stack([times, -offset[0], -offset[1], zeros])
    trajectory_b = np.column_stack([times, offset[0], offset[1], zeros])
    return trajectory_a, trajectory_b


def test_eccentricity_control_params():
    initial_separation = 16.0
    # Angular velocity at t = 0, see 'binary_trajectories' above
    angular_velocity = initial_separation**-1.5
    times = np.arange(0.0, 1500.0, 1.0)
    trajectory_a, trajectory_b = binary_trajectories(times, initial_separation)

    ecc_params = eccentricity_control_params(
        trajectory_a=trajectory_a,
        trajectory_b=trajectory_b,
        separation=initial_separation,
        orbital_angular_velocity=angular_velocity,
        radial_expansion_velocity=-1.0e-6,
        mass_a=0.5,
        mass_b=0.5,
        tmin=0.0,
        tmax=1200.0,
    )

    # The trajectories are circular by construction
    assert abs(ecc_params["Eccentricity"]) < 1.0e-5
    assert ecc_params["Omega0"] == angular_velocity
    assert ecc_params["Adot0"] == -1.0e-6
    assert ecc_params["D0"] == initial_separation
    assert ecc_params["Tmin"] == 0.0
    assert ecc_params["Tmax"] == 1200.0
    assert ecc_params["NewOmega0"] == (
        ecc_params["Omega0"] + ecc_params["DeltaOmega0"]
    )
    assert ecc_params["NewAdot0"] == (
        ecc_params["Adot0"] + ecc_params["DeltaAdot0"]
    )
    assert ecc_params["NewD0"] == ecc_params["D0"] + ecc_params["DeltaD0"]
