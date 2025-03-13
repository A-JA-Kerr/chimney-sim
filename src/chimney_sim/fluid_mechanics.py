import numpy as np

from .units import ureg, Q_
from .thermo_properties import density_amb

def friction_factor(Re):
    """
    Calculate the friction factor in a straight chimney section.

    The function determines the Darcy-Weisbach friction factor based on the Reynolds number (Re).
    It uses different correlations for laminar, transitional, and turbulent flow regimes.

    Parameters
    ----------
    Re : float
        Reynolds number (dimensionless).

    Returns
    -------
    float
        Friction factor (dimensionless), used to calculate pressure drop.
    """
    if Re <= 2000.0:  # Laminar flow
        fric_factor = 64.0 / Re
    elif 2000.0 < Re < 2600.0:  # Transitional flow
        gamma = (Re - 2000.0) / 600.0  # Weighting factor
        lam = 64.0 / 2000.0  # Laminar boundary
        turb = 0.3164 / (2600.0 ** 0.25)  # Turbulent boundary
        fric_factor = (1 - gamma) * lam + gamma * turb
    else:  # Turbulent flow
        fric_factor = 0.3164 / (Re ** 0.25)

    return fric_factor

def section_pressure_drop(h, D, v, Re, rho):
    """
    Calculate pressure drop in a straight chimney section according to Bernoulli's principle.

    Parameters
    ----------
    h : pint.Quantity
        Section height (in meters).
    D : pint.Quantity
        Inside diameter of the section (in meters).
    v : pint.Quantity
        Average flue gas velocity (in meters per second).
    Re : float
        Reynolds number (dimensionless).
    rho : pint.Quantity
        Average flue gas density (in kg/m³).

    Returns
    -------
    pint.Quantity
        Section frictional pressure drop (in Pascals).
    """
    dP = (rho*(v**2)*friction_factor(Re)*h)/(2*D)
    return dP.to('Pa')

def reynolds(rho, v, D, mu):
    """
    Calculate Reynolds number for flow inside the chimney.

    Parameters
    ----------
    rho : pint.Quantity
        Fluid density (in kg/m³).
    v : pint.Quantity
        Flow velocity (in meters per second).
    D : pint.Quantity
        Characteristic length (diameter of the chimney) (in meters).
    mu : pint.Quantity
        Dynamic viscosity (in Pa·s or kg/(m·s)).

    Returns
    -------
    pint.Quantity
        Reynolds number (dimensionless).
    """
    Rey = rho*v*D/mu
    return Rey

def wind_pressure(v_wind, T_amb, Pres):
    """
    Calculate the pressure contribution due to wind at the open flue end.

    Parameters
    ----------
    v_wind : pint.Quantity
        Wind velocity (in meters per second).
    T_amb : pint.Quantity
        Ambient temperature (in Kelvin).
    Pres : pint.Quantity
        Ambient pressure (in Pascals).

    Returns
    -------
    pint.Quantity
        Wind pressure contribution (in Pascals).
    """
    p_wind = 0.5 * density_amb(T_amb, Pres) * v_wind**2
    return p_wind