import numpy as np

from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity

def flue_velocity_in(V_dot, A_cs):
    """
    Compute the flue gas velocity using volumetric flow rate and cross-sectional area.

    The velocity is determined using the continuity equation:

        v_flue = V_dot / A_cs

    Assumptions:
    - Steady-state, incompressible flow.
    - Uniform velocity profile across the cross-section.

    Parameters
    ----------
    V_dot : pint.Quantity
        Volumetric flow rate of flue gas [m³/s].
    A_cs : pint.Quantity
        Cross-sectional area of the flue [m²].

    Returns
    -------
    pint.Quantity
        Flue gas velocity [m/s].
    """
    v_flue = V_dot / A_cs
    return v_flue.to('m/s')


def flue_velocity_general(P1, P2, rho_in, v_in, rho_out, g, h2):
    """
    Compute flue gas velocity using the full Bernoulli equation.

    The full Bernoulli equation:

        P1 + (1/2) * rho_in * v_in^2 + rho_out * g * h2 =
        P2 + (1/2) * rho_out * v_out^2

    Solving for v_out:

        v_out = sqrt( 2 * (P1 + (1/2) * rho_in * v_in^2 - P2 - rho_out * g * h2) / rho_out )

    Assumptions:
    - Incompressible, steady-state flow.
    - No significant viscous losses.

    Parameters
    ----------
    P1 : pint.Quantity
        Static pressure at the flue inlet [Pa].
    P2 : pint.Quantity
        Static pressure at the Pitot tube location [Pa].
    rho_in : pint.Quantity
        Density of the flue gas at the inlet [kg/m³].
    v_in : pint.Quantity
        Velocity of the flue gas at the inlet [m/s].
    rho_out : pint.Quantity
        Density of the flue gas at the Pitot tube location [kg/m³].
    g : pint.Quantity
        Gravitational acceleration [m/s²].
    h2 : pint.Quantity
        Height difference from the inlet to the Pitot tube [m].

    Returns
    -------
    pint.Quantity
        Velocity of the flue gas at the Pitot tube location [m/s].
    """
    term = (P1 + (0.5 * rho_in * v_in**2) - P2 - rho_out * g * h2) * (2 / rho_out)
    if term.magnitude < 0.0:
        term = Q_(0.0, 'm**2/s**2')
    return np.sqrt(term).to('m/s')


def flue_velocity_ignore_draft(P1, rho_in, v_in, rho_out, g, h2):
    """
    Compute flue gas velocity assuming negligible draft above the Pitot tube (P2 ≈ 0).

    The Bernoulli equation simplifies to:

        P1 + (1/2) * rho_in * v_in^2 + rho_out * g * h2 =
        (1/2) * rho_out * v_out^2

    Parameters
    ----------
    P1 : pint.Quantity
        Static pressure at the flue inlet [Pa].
    rho_in : pint.Quantity
        Density of the flue gas at the inlet [kg/m³].
    v_in : pint.Quantity
        Velocity of the flue gas at the inlet [m/s].
    rho_out : pint.Quantity
        Density of the flue gas at the Pitot tube location [kg/m³].
    g : pint.Quantity
        Gravitational acceleration [m/s²].
    h2 : pint.Quantity
        Height difference from the inlet to the Pitot tube [m].

    Returns
    -------
    pint.Quantity
        Velocity of the flue gas at the Pitot tube location [m/s].
    """
    term = (P1 + (0.5 * rho_in * v_in**2) - rho_out * g * h2) * (2 / rho_out)
    if term.magnitude < 0.0:
        term = Q_(0.0, 'm**2/s**2')
    return np.sqrt(term).to('m/s')


def flue_velocity_ignore_gravity(P1, P2, rho_in, v_in, rho_out):
    """
    Compute flue gas velocity assuming negligible gravitational effects (g * h2 ≈ 0).

    The Bernoulli equation simplifies to:

        P1 + (1/2) * rho_in * v_in^2 = P2 + (1/2) * rho_out * v_out^2

    Parameters
    ----------
    P1 : pint.Quantity
        Static pressure at the flue inlet [Pa].
    P2 : pint.Quantity
        Static pressure at the Pitot tube location [Pa].
    rho_in : pint.Quantity
        Density of the flue gas at the inlet [kg/m³].
    v_in : pint.Quantity
        Velocity of the flue gas at the inlet [m/s].
    rho_out : pint.Quantity
        Density of the flue gas at the Pitot tube location [kg/m³].

    Returns
    -------
    pint.Quantity
        Velocity of the flue gas at the Pitot tube location [m/s].
    """
    term = (P1 + (0.5 * rho_in * v_in**2) - P2) * (2 / rho_out)
    if term.magnitude < 0.0:
        term = Q_(0.0, 'm**2/s**2')
    return np.sqrt(term).to('m/s')


def flue_velocity_assume_static(P1, P2, rho_out, g, h2):
    """
    Compute flue gas velocity assuming the inlet velocity is zero (v_in ≈ 0).

    The Bernoulli equation simplifies to:

        P1 - P2 - rho_out * g * h2 = (1/2) * rho_out * v_out^2

    Parameters
    ----------
    P1 : pint.Quantity
        Static pressure at the flue inlet [Pa].
    P2 : pint.Quantity
        Static pressure at the Pitot tube location [Pa].
    rho_out : pint.Quantity
        Density of the flue gas at the Pitot tube location [kg/m³].
    g : pint.Quantity
        Gravitational acceleration [m/s²].
    h2 : pint.Quantity
        Height difference from the inlet to the Pitot tube [m].

    Returns
    -------
    pint.Quantity
        Velocity of the flue gas at the Pitot tube location [m/s].
    """
    term = (P1 - P2 - rho_out * g * h2) * (2 / rho_out)
    return np.sqrt(term).to('m/s')


def flue_velocity_static_nograv(P1, P2, rho_out):
    """
    Compute flue gas velocity assuming inlet velocity is zero and gravity is negligible.

    The Bernoulli equation simplifies to:

        P1 - P2 = (1/2) * rho_out * v_out^2

    Parameters
    ----------
    P1 : pint.Quantity
        Static pressure at the flue inlet [Pa].
    P2 : pint.Quantity
        Static pressure at the Pitot tube location [Pa].
    rho_out : pint.Quantity
        Density of the flue gas at the Pitot tube location [kg/m³].

    Returns
    -------
    pint.Quantity
        Velocity of the flue gas at the Pitot tube location [m/s].
    """
    term = (P1 - P2) * (2 / rho_out)
    return np.sqrt(term).to('m/s')
