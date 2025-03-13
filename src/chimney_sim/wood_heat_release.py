import numpy as np

from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity

def mdot_highpow(t):
    """
    Compute the mass flow rate of wood fuel using a fitted analytical derivative.

    The function is based on mass scale data from LBNL and computes the fuel mass 
    flow rate from the analytical derivative of the best-fit function for mass.

    Parameters
    ----------
    t : float
        Time [s].

    Returns
    -------
    pint.Quantity
        Mass flow rate of wood fuel [kg/s] at the given time.
    """
    m_dot = (343.0/238814)*np.exp(-1.0*50*(t-122763/50)/119407)
    return Q_(m_dot, 'kg/s')

def mass_ratio_products(lambda_excess_air):
    """
    Compute the total mass of combustion products and air per unit mass of fuel.

    This function calculates the total mass of combustion products (including 
    excess air) per unit mass of cellulose fuel, assuming complete combustion.

    Parameters
    ----------
    lambda_excess_air : float
        Excess air ratio (lambda), where lambda > 1.

    Returns
    -------
    float
        Mass of combustion products per unit mass of cellulose.
    """
    return 1.0 + 5.089 * lambda_excess_air