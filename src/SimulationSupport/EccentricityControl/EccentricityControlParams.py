# Distributed under the MIT License.
# See LICENSE.txt for details.
"""Estimate eccentricity and updated orbital parameters from trajectories."""

import logging
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import numpy as np

from .OmegaDotEccRemoval import (
    ComputeOmegaAndDerivsFromFile,
    FindTmin,
    performAllFits,
)

logger = logging.getLogger(__name__)

# Keys of the dictionary returned by 'eccentricity_control_params'
EccentricityParams = Literal[
    "Eccentricity",
    "EccentricityError",
    "Omega0",
    "Adot0",
    "D0",
    "DeltaOmega0",
    "DeltaAdot0",
    "DeltaD0",
    "NewOmega0",
    "NewAdot0",
    "NewD0",
    "Tmin",
    "Tmax",
]


def eccentricity_control_params(
    trajectory_a: np.ndarray,
    trajectory_b: np.ndarray,
    separation: float,
    orbital_angular_velocity: float,
    radial_expansion_velocity: float,
    mass_a: float,
    mass_b: float,
    spin_a: Optional[np.ndarray] = None,
    spin_b: Optional[np.ndarray] = None,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    target_eccentricity: float = 0.0,
    plot_output_dir: Optional[Union[str, Path]] = None,
) -> Dict[EccentricityParams, float]:
    r"""Get new orbital parameters for a binary system to control eccentricity.

    The eccentricity is estimated from the trajectories of the binary objects
    and updates to the orbital parameters are suggested to drive the orbit to
    the target eccentricity, using SpEC's ``OmegaDotEccRemoval.py``. Currently
    supports only circular target orbits (target eccentricity = 0).

    Parameters
    ----------
    trajectory_a : numpy.ndarray
        Trajectory of the first object, with shape ``(num_times, 4)``. The
        first column is the time, the remaining three columns are the
        coordinates.
    trajectory_b : numpy.ndarray
        Trajectory of the second object, in the same format as
        ``trajectory_a``.
    separation : float
        Initial coordinate separation ``D_0`` of the two objects, i.e. the
        initial data parameter that is being controlled.
    orbital_angular_velocity : float
        Initial orbital angular velocity ``Omega_0``.
    radial_expansion_velocity : float
        Initial radial expansion velocity ``adot_0``.
    mass_a : float
        Christodoulou mass of the first object at the reference time.
    mass_b : float
        Christodoulou mass of the second object at the reference time.
    spin_a : numpy.ndarray, optional
        Dimensionful spin of the first object, with shape ``(num_times, 4)``.
        The first column is the time, the remaining three columns are the
        components of the spin vector. If either spin is unspecified, spin
        effects are ignored in the fits.
    spin_b : numpy.ndarray, optional
        Dimensionful spin of the second object, in the same format as
        ``spin_a``.
    tmin : float, optional
        The lower time bound for the eccentricity estimate. Used to remove
        initial junk and transients in the data. If unspecified, uses SpEC's
        ``OmegaDotEccRemoval.FindTmin`` to estimate it.
    tmax : float, optional
        The upper time bound for the eccentricity estimate. A reasonable value
        would include 2-3 orbits. Default is ``500 + 5 * pi / Omega_0``.
    target_eccentricity : float, optional
        Eccentricity that the updated orbital parameters should achieve.
        Currently only 0 (circular orbit) is supported.
    plot_output_dir : str or pathlib.Path, optional
        Output directory for plots.

    Returns
    -------
    dict
        Dictionary with the keys listed in ``EccentricityParams``.
    """
    if target_eccentricity != 0.0:
        raise ValueError(
            "Only circular orbits are currently supported for eccentricity"
            " control."
        )

    # Compute the orbital frequency and its time derivative from the
    # trajectories
    t, Omega, dOmegadt, OmegaVec = ComputeOmegaAndDerivsFromFile(
        trajectory_a, trajectory_b
    )

    # Set time bounds if not provided
    if tmin is None:
        tmin = max(FindTmin(t, dOmegadt, 500), t[0])
    if tmax is None:
        tmax = min(500 + 5 * np.pi / orbital_angular_velocity, t[-1])
    logger.info(
        "Estimating eccentricity from trajectory data in time range"
        f" {tmin:.3f} to {tmax:.3f}."
    )

    # Call into SpEC's OmegaDotEccRemoval.py
    eccentricity, delta_Omega0, delta_adot0, delta_D0, ecc_std_dev, _ = (
        performAllFits(
            XA=trajectory_a,
            XB=trajectory_b,
            t=t,
            Omega=Omega,
            dOmegadt=dOmegadt,
            OmegaVec=OmegaVec,
            mA=mass_a,
            mB=mass_b,
            sA=spin_a,
            sB=spin_b,
            IDparam_omega0=orbital_angular_velocity,
            IDparam_adot0=radial_expansion_velocity,
            IDparam_D0=separation,
            tmin=tmin,
            tmax=tmax,
            tref=tmin,
            opt_freq_filter=True,
            opt_varpro=True,
            opt_type="bbh",
            opt_tmin=tmin,
            opt_improved_Omega0_update=True,
            check_periastron_advance=True,
            plot_output_dir=plot_output_dir,
            Source="",
        )
    )
    logger.info(
        f"Eccentricity estimate is {eccentricity:g} +/- {ecc_std_dev:e}."
        " Update orbital parameters as follows"
        f" for target eccentricity {target_eccentricity:g} (choose two):\n"
        f"Omega0 += {delta_Omega0:e} ->"
        f" {orbital_angular_velocity + delta_Omega0:.8g}\n"
        f"adot0 += {delta_adot0:e} ->"
        f" {radial_expansion_velocity + delta_adot0:e}\n"
        f"D0 += {delta_D0:e} -> {separation + delta_D0:.8g}"
    )
    # These keys must correspond to 'EccentricityParams'
    return {
        "Eccentricity": eccentricity,
        "EccentricityError": ecc_std_dev,
        "Omega0": orbital_angular_velocity,
        "Adot0": radial_expansion_velocity,
        "D0": separation,
        "DeltaOmega0": delta_Omega0,
        "DeltaAdot0": delta_adot0,
        "DeltaD0": delta_D0,
        "NewOmega0": orbital_angular_velocity + delta_Omega0,
        "NewAdot0": radial_expansion_velocity + delta_adot0,
        "NewD0": separation + delta_D0,
        "Tmin": tmin,
        "Tmax": tmax,
    }
