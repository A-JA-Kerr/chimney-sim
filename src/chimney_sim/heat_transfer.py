import numpy as np

from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity

def section_stovepipe_mass(diameter_inner, height, thickness, density):
    """
    Compute the mass of the steel in a stovepipe section.

    Parameters
    ----------
    diameter_inner : pint.Quantity
        Inner diameter of the stovepipe [m].
    height : pint.Quantity
        Section height [m].
    thickness : pint.Quantity
        Wall thickness of the steel pipe [m].
    density : pint.Quantity
        Density of the steel material [kg/m³].

    Returns
    -------
    pint.Quantity
        Mass of the steel in the stovepipe section [kg].
    """
    m_steel = thickness*diameter_inner*np.pi*height*density
    return m_steel

def section_insulation_mass(diameter_inner, diameter_outer, height, density):
    """
    Compute the mass of the insulation in a stovepipe section.

    Parameters
    ----------
    diameter_inner : pint.Quantity
        Inner diameter of the insulation layer [m].
    diameter_outer : pint.Quantity
        Outer diameter of the insulation layer [m].
    height : pint.Quantity
        Section height [m].
    density : pint.Quantity
        Density of the insulation material [kg/m³].

    Returns
    -------
    pint.Quantity
        Mass of the insulation in the stovepipe section [kg].
    """
    m_iso = 0.5*(diameter_outer-diameter_inner)*diameter_outer*np.pi*height*density
    return m_iso

def conductivity_iso_krueger(T_iso):
    """
    Compute the thermal conductivity of insulation using Krueger's correlation.

    Parameters
    ----------
    T_iso : pint.Quantity
        Temperature of the insulation material [°C].

    Returns
    -------
    pint.Quantity
        Thermal conductivity of the insulation material [W/(m·K)].
    """
    k_iso = 4.81481481*(10**-7)*(T_iso.to('degC').magnitude**2) + 1.91481481*(10**-4)*T_iso.to('degC').magnitude + 3.6037037*(10**-2)
    return Q_(k_iso, 'W/(m*K)')

def inner_heat_transfer_coefficient(diameter, velocity):
    """
    Compute the inner convective heat transfer coefficient for chimney flow.

    Parameters
    ----------
    diameter : pint.Quantity
        Inner pipe diameter [m].
    velocity : pint.Quantity
        Flow velocity of flue gases [m/s].

    Returns
    -------
    pint.Quantity
        Inner heat transfer coefficient [W/(m²·K)].
    """
    htc = 4.4 * (velocity.to('m/s').magnitude**0.75) / (diameter.to('m').magnitude**0.25)
    return Q_(htc, 'W/(m**2*K)')


def outer_heat_transfer_coefficient(diameter, wind_velocity, T_steel, T_iso, T_o, h):
    """
    Compute the outer heat transfer coefficient based on natural and forced convection.

    The function determines the dominant heat transfer mode by comparing
    natural convection due to surface temperature differences and forced 
    convection from wind velocity.

    Parameters
    ----------
    diameter : pint.Quantity
        Outer pipe diameter [m].
    wind_velocity : pint.Quantity
        Wind velocity outside the chimney [m/s].
    T_steel : pint.Quantity
        Stovepipe wall temperature [K].
    T_iso : pint.Quantity
        Insulation surface temperature [K].
    T_o : pint.Quantity
        Outdoor ambient temperature [K].
    h : pint.Quantity
        Chimney section height [m].

    Returns
    -------
    pint.Quantity
        Outer heat transfer coefficient [W/(m²·K)].
    """
    nat_htc = 1.32 * ((dT_surface(T_steel, T_iso, T_o).to('K').magnitude / h.to('m').magnitude) ** 0.25)
    forced_htc = (8.9 * (wind_velocity.to('m/s').magnitude) ** 0.9) / (diameter.to('m').magnitude ** 0.1)
    htc = max(nat_htc, forced_htc)
    return Q_(htc, 'W/(m**2*K)')


def dT_surface(T_steel, T_iso, T_o):
    """
    Compute the effective temperature difference driving outside convection.

    Parameters
    ----------
    T_steel : pint.Quantity
        Stovepipe wall temperature [K].
    T_iso : pint.Quantity
        Insulation surface temperature [K].
    T_o : pint.Quantity
        Outdoor ambient temperature [K].

    Returns
    -------
    pint.Quantity
        Effective temperature difference [K].
    """
    dT = T_steel - 2.0 * (T_steel - T_iso) - T_o
    return dT


def heat_transmission_in(h, htc_conv_inside, d_i):
    """
    Compute the heat transmission coefficient from flue gas to the chimney wall.

    This accounts for the geometry of the chimney and is distinct from 
    the convective heat transfer coefficient.

    Parameters
    ----------
    h : pint.Quantity
        Chimney section height [m].
    htc_conv_inside : pint.Quantity
        Heat transfer coefficient between flue gas and chimney surface [W/(m²·K)].
    d_i : pint.Quantity
        Inner diameter of the chimney pipe [m].

    Returns
    -------
    pint.Quantity
        Heat transmission coefficient from flue gas to chimney wall [W/K].
    """
    kA_in = (2.0 * np.pi * h) / (1.0 / (htc_conv_inside * d_i / 2.0))
    return kA_in


def heat_transmission_through(h, d_o, d_i, K_iso, htc_conv_outer):
    """
    Compute the heat transmission coefficient through chimney insulation.

    This accounts for both conductive heat transfer through the insulation layer
    and convective heat transfer at the outer surface.

    Parameters
    ----------
    h : pint.Quantity
        Chimney section height [m].
    d_o : pint.Quantity
        Outer diameter of the insulation layer [m].
    d_i : pint.Quantity
        Inner diameter of the insulation layer [m].
    K_iso : pint.Quantity
        Thermal conductivity of the insulation material [W/(m·K)].
    htc_conv_outer : pint.Quantity
        Outer convective heat transfer coefficient [W/(m²·K)].

    Returns
    -------
    pint.Quantity
        Heat transmission coefficient through insulation [W/K].
    """
    numer = 2.0 * np.pi * h
    denom = np.log(d_o / d_i) / K_iso + 1 / (htc_conv_outer * d_o / 2.0)
    kA_through = numer / denom
    return kA_through


def heat_transmission_out(h, htc_conv_outside, d_o):
    """
    Compute the heat transmission coefficient from the chimney wall to the environment.

    This function is identical to `heat_transmission_in()` and may be consolidated in future.

    Parameters
    ----------
    h : pint.Quantity
        Chimney section height [m].
    htc_conv_outside : pint.Quantity
        Heat transfer coefficient between the chimney outer surface and ambient air [W/(m²·K)].
    d_o : pint.Quantity
        Outer diameter of the chimney pipe [m].

    Returns
    -------
    pint.Quantity
        Heat transmission coefficient from chimney wall to surroundings [W/K].
    """
    kA_out = (2.0 * np.pi * h) / (1.0 / (htc_conv_outside * d_o / 2.0))
    return kA_out

def calculate_T_flue_out(gamma_1, h, T_flue_in, T_steel):
    """
    Compute the temperature of flue gas leaving a chimney section.

    Parameters
    ----------
    gamma_1 : pint.Quantity
        Intermediate calculation value for heat transfer in the flue [1/m].
    h : pint.Quantity
        Chimney section height [m].
    T_flue_in : pint.Quantity
        Inlet flue gas temperature [K].
    T_steel : pint.Quantity
        Steel wall temperature of the chimney section [K].

    Returns
    -------
    pint.Quantity
        Flue gas exit temperature [K].
    """
    T_flue_out = np.exp(gamma_1 * h) * (T_flue_in - T_steel) + T_steel
    return T_flue_out.to('K')


def intermediate_gamma_1(kA_1, h, m_dot_flue, C_flue):
    """
    Compute the intermediate value gamma_1 for flue heat transfer.

    Parameters
    ----------
    kA_1 : pint.Quantity
        Internal heat transmission coefficient [W/K].
    h : pint.Quantity
        Chimney section height [m].
    m_dot_flue : pint.Quantity
        Mass flow rate of flue gases [kg/s].
    C_flue : pint.Quantity
        Heat capacity of flue gas [J/(kg·K)].

    Returns
    -------
    pint.Quantity
        Intermediate heat transfer coefficient gamma_1 [1/m].
    """
    gamma_1 = -1.0 * kA_1 / (h * m_dot_flue * C_flue)
    return gamma_1.to('m**-1')


def new_T_flue_av(T_flue_in, T_steel, gamma_1, h):
    """
    Compute the updated average flue gas temperature for iterative solutions.

    Parameters
    ----------
    T_flue_in : pint.Quantity
        Inlet flue gas temperature [K].
    T_steel : pint.Quantity
        Steel wall temperature of the chimney section [K].
    gamma_1 : pint.Quantity
        Intermediate heat transfer coefficient [1/m].
    h : pint.Quantity
        Chimney section height [m].

    Returns
    -------
    pint.Quantity
        Updated average flue gas temperature [K].
    """
    T_av_new = (T_flue_in - T_steel) * (np.exp(gamma_1 * h) - 1.0) / (gamma_1 * h) + T_steel
    return T_av_new.to('K')


def intermediate_gamma_2(m_steel, C_steel, m_iso, C_iso, kA_2, kA_3):
    """
    Compute the intermediate value gamma_2 for chimney heat transfer.

    Parameters
    ----------
    m_steel : pint.Quantity
        Mass of the steel chimney wall [kg].
    C_steel : pint.Quantity
        Heat capacity of the steel [J/(kg·K)].
    m_iso : pint.Quantity
        Mass of the insulation material [kg].
    C_iso : pint.Quantity
        Heat capacity of the insulation material [J/(kg·K)].
    kA_2 : pint.Quantity
        Heat transmission coefficient through the chimney wall [W/K].
    kA_3 : pint.Quantity
        Heat transmission coefficient to the surroundings [W/K].

    Returns
    -------
    pint.Quantity
        Intermediate heat capacity term gamma_2 [J/K].
    """
    gamma_2 = (m_steel * C_steel + 0.5 * m_iso * C_iso) * (1.0 + kA_2 / kA_3)
    return gamma_2.to('J/K')


def new_T_steel(kA_1, T_flue_av, T_steel, kA_2, T_o, gamma_2):
    """
    Compute the updated steel temperature for iterative solutions.

    This follows Equation (12) from Krueger.

    Parameters
    ----------
    kA_1 : pint.Quantity
        Heat transmission coefficient between flue gas and steel wall [W/K].
    T_flue_av : pint.Quantity
        Average flue gas temperature in the section [K].
    T_steel : pint.Quantity
        Steel wall temperature of the chimney section [K].
    kA_2 : pint.Quantity
        Heat transmission coefficient through the chimney wall [W/K].
    T_o : pint.Quantity
        Outdoor ambient temperature [K].
    gamma_2 : pint.Quantity
        Intermediate heat capacity term [J/K].

    Returns
    -------
    pint.Quantity
        Updated steel temperature [K].
    """
    T_steel_new = T_steel.to('K').magnitude + (1.0 / gamma_2.to('J/K').magnitude) * \
        (kA_1.to('W/K').magnitude * (T_flue_av.to('K').magnitude - T_steel.to('K').magnitude) -
         kA_2.to('W/K').magnitude * (T_steel.to('K').magnitude - T_o.to('K').magnitude))
    
    return Q_(T_steel_new, 'K')


def new_T_iso(kA_2, kA_3, T_steel_new, T_o):
    """
    Compute the updated insulation temperature for iterative solutions.

    Parameters
    ----------
    kA_2 : pint.Quantity
        Heat transmission coefficient through the chimney wall [W/K].
    kA_3 : pint.Quantity
        Heat transmission coefficient between chimney and outdoors [W/K].
    T_steel_new : pint.Quantity
        Updated steel wall temperature from the iterative solution [K].
    T_o : pint.Quantity
        Outdoor ambient temperature [K].

    Returns
    -------
    pint.Quantity
        Updated insulation temperature [K].
    """
    T_iso_new = 0.5 * ((1.0 + kA_2 / kA_3) * T_steel_new + (1.0 - kA_2 / kA_3) * T_o)
    return T_iso_new.to('K')

def iterate_flue_gas_temperature(T_flue_in, T_steel, T_iso, T_o, h, kA_1, kA_2, kA_3, gamma_1, gamma_2, tol=1e-3, max_iter=30):
    """
    Iteratively solve for the average flue gas temperature using an implicit method.

    This function iterates to find a self-consistent solution for the flue gas,
    steel wall, and insulation temperatures in a chimney section, based on 
    heat transfer between the components.

    Parameters
    ----------
    T_flue_in : pint.Quantity
        Initial flue gas inlet temperature [K].
    T_steel : pint.Quantity
        Initial temperature of the steel wall [K].
    T_iso : pint.Quantity
        Initial temperature of the insulation layer [K].
    T_o : pint.Quantity
        Ambient temperature [K].
    h : pint.Quantity
        Height of the chimney section [m].
    kA_1 : pint.Quantity
        Heat transmission coefficient from gas to steel [W/K].
    kA_2 : pint.Quantity
        Heat transmission coefficient through the steel wall [W/K].
    kA_3 : pint.Quantity
        Heat transmission coefficient from steel to ambient [W/K].
    gamma_1 : pint.Quantity
        Heat loss factor for flue gas temperature decay [1/m].
    gamma_2 : pint.Quantity
        Heat capacity factor for steel and insulation [J/K].
    tol : float, optional
        Convergence tolerance for the flue gas temperature [K]. Default is 1e-3 (1 mK).
    max_iter : int, optional
        Maximum number of iterations. Default is 30.

    Returns
    -------
    tuple of pint.Quantity
        - Updated average flue gas temperature [K].
        - Updated steel temperature [K].
        - Updated insulation temperature [K].

    Warns
    -----
    UserWarning
        If the method does not converge within the maximum number of iterations.
    """
    T_flue_av = T_flue_in  # Initial guess

    for i in range(max_iter):  # Limit iterations to prevent infinite loops
        T_flue_av_new = new_T_flue_av(T_flue_in, T_steel, gamma_1, h)
        T_steel_new = new_T_steel(kA_1, T_flue_av, T_steel, kA_2, T_o, gamma_2)
        T_iso_new = new_T_iso(kA_2, kA_3, T_steel_new, T_o)

        # Check convergence on the flue gas temperature
        if abs(T_flue_av_new.to('K').magnitude - T_flue_av.to('K').magnitude) < tol:
            return T_flue_av_new, T_steel_new, T_iso_new

        # Update variables for next iteration
        T_flue_av = T_flue_av_new
        T_steel = T_steel_new
        T_iso = T_iso_new

    # If not converged, warn the user
    print(f"Warning: iterate_flue_gas_temperature did not converge within {max_iter} iterations.")
    return T_flue_av_new, T_steel_new, T_iso_new