import numpy as np

from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity

def Cp_steel():
    """
    Return the specific heat capacity of AISI 1010 steel at 300 K.

    This function currently returns a constant value but could be extended to
    include temperature-dependent variations.

    Returns
    -------
    pint.Quantity
        Specific heat capacity of AISI 1010 steel [J/(kg·K)].
    """
    return Q_(434.0, 'J/(kg*K)')


def rho_iso():
    """
    Return the density of mineral wool insulation.

    The value is taken from Table A.3 of *Fundamentals of Heat and Mass Transfer*.

    Returns
    -------
    pint.Quantity
        Density of mineral wool granules [kg/m³].
    """
    return Q_(190.0, 'kg/(m**3)')


def Cp_iso():
    """
    Return the specific heat capacity of mineral wool insulation.

    The value is taken from Table A.3 of *Fundamentals of Heat and Mass Transfer*.

    Returns
    -------
    pint.Quantity
        Specific heat capacity of mineral wool granules [J/(kg·K)].
    """
    return Q_(835.0, 'J/(kg*K)')

def density_flue_gas(T):
    """
    Compute the density of flue gas using the ideal gas law.

    Assumes a constant molar mass and standard atmospheric pressure.

    Parameters
    ----------
    T : pint.Quantity
        Temperature of the flue gas [K].

    Returns
    -------
    pint.Quantity
        Density of the flue gas [kg/m³].
    """
    T_K = T.to('K').magnitude
    P = 101325  # Pressure in Pa (standard atmospheric pressure)
    M = 0.029559  # Molar mass of flue gas [kg/mol]
    R = 8.314  # Universal gas constant [J/(mol·K)]

    rho = (P * M) / (R * T_K)
    return Q_(rho, 'kg/m**3')


def density_amb(T):
    """
    Compute the density of ambient air using the ideal gas law.

    Assumes a constant molar mass and standard atmospheric pressure.

    Parameters
    ----------
    T : pint.Quantity
        Temperature of the ambient air [K].

    Returns
    -------
    pint.Quantity
        Density of the ambient air [kg/m³].
    """
    T_K = T.to('K').magnitude
    P = 101325  # Pressure in Pa (standard atmospheric pressure)
    M = 0.02897  # Molar mass of dry air [kg/mol]
    R = 8.314  # Universal gas constant [J/(mol·K)]

    rho = (P * M) / (R * T_K)
    return Q_(rho, 'kg/m**3')


def Cp_flue_gas(T):
    """
    Compute the specific heat capacity of flue gas.

    The calculation assumes an approximate molar composition of:
    - 70% N₂
    - 15% CO₂
    - 15% H₂O

    The heat capacity is determined using NIST Shomate equations, weighted by molar fraction.

    Parameters
    ----------
    T : pint.Quantity
        Temperature of the flue gas [K].

    Returns
    -------
    pint.Quantity
        Specific heat capacity of flue gas [J/(kg·K)].

    Notes
    -----
    - Shomate parameters are used for each gas component.
    - The function assumes a constant molar mass of 0.029559 kg/mol.
    """
    T_K = T.to('K').magnitude
    t = T_K / 1000  # Shomate temperature scaling

    # Shomate parameters (valid for T ≤ 500 K)
    shomate_params = {
        "N2":  (28.98641, 1.853978, -9.647459, 16.63537, 0.000117),
        "H2O": (30.09200, 6.832514, 6.793435, -2.534480, 0.082139),
        "CO2": (24.99735, 55.18696, -33.69137, 7.948387, -0.136638)
    }

    # Mole fractions
    mole_fractions = {"N2": 0.70, "H2O": 0.15, "CO2": 0.15}

    # Compute Cp for each species using Shomate equation
    def compute_cp(coeffs):
        A, B, C, D, E = coeffs
        return A + B * t + C * t**2 + D * t**3 + E / t**2

    # Weighted sum for mixture Cp
    Cp_mixture = sum(mole_fractions[gas] * compute_cp(shomate_params[gas]) for gas in mole_fractions)

    # Convert to mass basis using assumed molar mass (kg/mol)
    molar_mass = 0.029559  # kg/mol
    Cp_mass = Cp_mixture / molar_mass  # J/(kg·K)

    return Q_(Cp_mass, 'J/(kg*K)')


def mu_flue_gas(T):
    """
    Compute the dynamic viscosity of CO₂ in mPa·s as a function of temperature.

    The viscosity is calculated using the NIST correlation (Equation 3.2 and Table 3.1).

    Parameters
    ----------
    T : pint.Quantity
        Temperature of the flue gas [K].

    Returns
    -------
    pint.Quantity
        Dynamic viscosity of CO₂ [mPa·s].

    Notes
    -----
    - The equation is derived from the NIST reference correlation for CO₂ viscosity.
    - Valid for typical flue gas temperature ranges.
    """
    T_K = T.to('K').magnitude

    # Coefficients from NIST Table 3.1
    a = [
        1749.354893188350,
        -369.06930007128,
        5423856.34887691,
        -2.21238352168356,
        -269503.247933569,
        73145.021531826,
        5.34368649509278
    ]

    sqrt_T = np.sqrt(T_K)
    T16 = T_K ** (1/6)
    T13 = T_K ** (1/3)

    numerator = 1.0055 * sqrt_T
    denominator = (
        a[0] + a[1] * T16 + a[2] * np.exp(a[3] * T13) +
        (a[4] + a[5] * T13) / np.exp(T13) + a[6] * sqrt_T
    )

    eta = numerator / denominator  # Viscosity in mPa·s
    return Q_(eta, 'mPa*s')