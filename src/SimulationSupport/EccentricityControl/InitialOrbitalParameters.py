# Distributed under the MIT License.
# See LICENSE.txt for details.
"""Estimate initial orbital parameters."""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def initial_orbital_parameters(
    target_params: dict,
    separation: Optional[float] = None,
    orbital_angular_velocity: Optional[float] = None,
    radial_expansion_velocity: Optional[float] = None,
) -> Tuple[float, float, float]:
    r"""Estimate initial orbital parameters from a Post-Newtonian approximation.

    Given the target eccentricity and one other orbital parameter, this
    routine estimates the initial separation ``D``, orbital angular velocity
    ``Omega_0``, and radial expansion velocity ``adot_0`` for a binary system.
    The resulting parameters can be fed into an eccentricity control loop to
    refine the starting parameters.

    Parameters
    ----------
    target_params : dict
        Simulation parameters that describe the binary. The dictionary must
        include the following keys:

        * ``"MassRatio"``: Mass ratio :math:`q = M_A / M_B \ge 1`.
        * ``"DimensionlessSpinA"``: Dimensionless spin vector of the larger
          black hole (length 3).
        * ``"DimensionlessSpinB"``: Dimensionless spin vector of the smaller
          black hole (length 3).
        * ``"Eccentricity"``: Target orbital eccentricity. Provide this value
          together with exactly one of the orbital parameters below.

        Optional keys:

        * ``"MeanAnomalyFraction"``: Mean anomaly divided by :math:`2\pi`
          (between 0 and 1). Required for nonzero eccentricity.
        * ``"NumOrbits"``: Desired number of inspiral orbits until merger.
        * ``"TimeToMerger"``: Desired time to merger.
    separation : float, optional
        Coordinate separation ``D`` of the black holes.
    orbital_angular_velocity : float, optional
        Orbital angular velocity ``Omega_0``.
    radial_expansion_velocity : float, optional
        Radial expansion velocity ``adot_0``.

    Returns
    -------
    tuple[float, float, float]
        A tuple ``(D_0, Omega_0, adot_0)`` with the initial separation,
        orbital angular velocity, and radial expansion velocity.
    """
    # If all orbital parameters are already specified, return early
    if (
        separation is not None
        and orbital_angular_velocity is not None
        and radial_expansion_velocity is not None
    ):
        return separation, orbital_angular_velocity, radial_expansion_velocity

    mass_ratio = target_params["MassRatio"]
    dimensionless_spin_a = np.asarray(target_params["DimensionlessSpinA"])
    dimensionless_spin_b = np.asarray(target_params["DimensionlessSpinB"])
    eccentricity = target_params["Eccentricity"]
    mean_anomaly_fraction = target_params.get("MeanAnomalyFraction")
    num_orbits = target_params.get("NumOrbits")
    time_to_merger = target_params.get("TimeToMerger")

    # Check input parameters for consistency
    assert eccentricity is not None, (
        "Specify all orbital parameters 'separation',"
        " 'orbital_angular_velocity', and 'radial_expansion_velocity', or"
        " specify an 'eccentricity' plus one orbital parameter."
    )
    if eccentricity != 0.0:
        assert mean_anomaly_fraction is not None, (
            "If you specify a nonzero 'eccentricity' you must also specify a"
            " 'mean_anomaly_fraction'."
        )
    assert radial_expansion_velocity is None, (
        "Can't use the 'radial_expansion_velocity' to compute orbital"
        " parameters. Remove it and choose another orbital parameter."
    )
    assert (
        (separation is not None)
        ^ (orbital_angular_velocity is not None)
        ^ (num_orbits is not None)
        ^ (time_to_merger is not None)
    ), (
        "Specify an 'eccentricity' plus _one_ of the following orbital"
        " parameters: 'separation', 'orbital_angular_velocity', 'num_orbits',"
        " 'time_to_merger'."
    )

    # Import functions from SpEC. These functions currently work only for zero
    # eccentricity. We will need to generalize this for eccentric orbits.
    assert eccentricity == 0.0, (
        "Initial orbital parameters can currently only be computed for zero"
        " eccentricity."
    )
    # These functions call old Fortran code (LSODA) through
    # scipy.integrate.odeint, which leads to lots of noise in stdout. We should
    # modernize them to use scipy.integrate.solve_ivp.
    from .ZeroEccParamsFromPN import nOrbitsAndTotalTime, omegaAndAdot

    # Find an omega0 that gives the right number of orbits or time to merger
    if num_orbits is not None or time_to_merger is not None:
        opt_result = minimize(
            lambda x: (
                abs(
                    nOrbitsAndTotalTime(
                        q=mass_ratio,
                        chiA0=dimensionless_spin_a,
                        chiB0=dimensionless_spin_b,
                        omega0=x[0],
                    )[0 if num_orbits is not None else 1]
                    - (num_orbits if num_orbits is not None else time_to_merger)
                )
            ),
            x0=[0.01],
            method="Nelder-Mead",
        )
        if not opt_result.success:
            raise ValueError(
                "Failed to find an orbital angular velocity that gives the"
                " desired number of orbits or time to merger. Error:"
                f" {opt_result.message}"
            )
        orbital_angular_velocity = opt_result.x[0]
        logger.debug(
            f"Found orbital angular velocity: {orbital_angular_velocity}"
        )

    # Find the separation that gives the desired orbital angular velocity
    if orbital_angular_velocity is not None:
        opt_result = minimize(
            lambda x: abs(
                omegaAndAdot(
                    r=x[0],
                    q=mass_ratio,
                    chiA=dimensionless_spin_a,
                    chiB=dimensionless_spin_b,
                    rPrime0=1.0,  # Choice also made in SpEC
                )[0]
                - orbital_angular_velocity
            ),
            x0=[10.0],
            method="Nelder-Mead",
        )
        if not opt_result.success:
            raise ValueError(
                "Failed to find a separation that gives the desired orbital"
                f" angular velocity. Error: {opt_result.message}"
            )
        separation = opt_result.x[0]
        logger.debug(f"Found initial separation: {separation}")

    # Find the radial expansion velocity
    new_orbital_angular_velocity, radial_expansion_velocity = omegaAndAdot(
        r=separation,
        q=mass_ratio,
        chiA=dimensionless_spin_a,
        chiB=dimensionless_spin_b,
        rPrime0=1.0,  # Choice also made in SpEC
    )
    if orbital_angular_velocity is None:
        orbital_angular_velocity = new_orbital_angular_velocity
    else:
        assert np.isclose(
            new_orbital_angular_velocity, orbital_angular_velocity, rtol=1e-4
        ), (
            "Orbital angular velocity is inconsistent with separation."
            " Maybe the rootfind failed to reach sufficient accuracy."
        )

    # Estimate number of orbits and time to merger
    num_orbits, time_to_merger = nOrbitsAndTotalTime(
        q=mass_ratio,
        chiA0=dimensionless_spin_a,
        chiB0=dimensionless_spin_b,
        omega0=orbital_angular_velocity,
    )
    logger.info(
        "Selected approximately circular orbit. Number of orbits:"
        f" {num_orbits:g}. Time to merger: {time_to_merger:g} M."
    )
    return separation, orbital_angular_velocity, radial_expansion_velocity
