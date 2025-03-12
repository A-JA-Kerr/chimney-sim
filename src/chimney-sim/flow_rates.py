
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
