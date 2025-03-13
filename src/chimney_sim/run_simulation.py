import numpy as np

from .units import ureg, Q_

""" Relative Imports"""
from .wood_heat_release import mdot_highpow
from .wood_heat_release import mass_ratio_products

from .thermo_properties import density_flue_gas
from .thermo_properties import Cp_flue_gas
from .thermo_properties import Cp_steel
from .thermo_properties import Cp_iso
from .thermo_properties import density_amb
from .thermo_properties import mu_flue_gas

from .heat_transfer import inner_heat_transfer_coefficient
from .heat_transfer import outer_heat_transfer_coefficient
from .heat_transfer import heat_transmission_in
from .heat_transfer import heat_transmission_through
from .heat_transfer import heat_transmission_out
from .heat_transfer import intermediate_gamma_1
from .heat_transfer import intermediate_gamma_2
from .heat_transfer import iterate_flue_gas_temperature
from .heat_transfer import calculate_T_flue_out
from .heat_transfer import conductivity_iso_krueger

from .fluid_mechanics import reynolds
from .fluid_mechanics import section_pressure_drop
from .fluid_mechanics import wind_pressure

from .flow_rates import flue_velocity_in

from .geometry_setup import section_height_arr
from .geometry_setup import m_steel
from .geometry_setup import d_i_arr
from .geometry_setup import overall_height_arr
from .geometry_setup import A_cs
from .geometry_setup import d_o_arr
from .geometry_setup import m_iso


def run_simulation(t_simulation, timestep, T_amb, P_amb, v_wind, T_flue_inlet, excess_air):
    """
    Runs the chimney draft simulation with the given inputs.

    Parameters
    ----------
    t_simulation : float
        Total simulation time in seconds.
    timestep : float
        Time step for simulation in seconds.
    T_amb : pint.Quantity
        Ambient temperature.
    P_amb : pint.Quantity
        Ambient pressure.
    v_wind : pint.Quantity
        Wind velocity.
    T_flue_inlet : pint.Quantity
        Flue gas inlet temperature.
    excess_air : float
        Excess air ratio.

    Returns
    -------
    tuple
        (times, T_flue_time, vol_flow_rates_out, P_total_time)
        - times : np.ndarray
            Time steps.
        - T_flue_time : np.ndarray
            Flue gas temperature profile over time.
        - vol_flow_rates_out : np.ndarray
            Volume flow rates at the outlet over time.
        - P_total_time : np.ndarray
            Total pressure over time.
    """

    # Gravitational acceleration
    grav = Q_(9.8, 'm/s**2')

    ## Simulation time array
    times = np.arange(0.0, t_simulation, timestep)

    ## Allocate Time-Dependent Arrays
    T_flue_time = np.zeros(len(times), dtype=object) # Use dtype=object for quantity arrays
    T_steel_time = np.zeros(len(times), dtype=object) 
    T_iso_time = np.zeros(len(times), dtype=object) 
    P_buoyant_time = Q_(np.zeros(len(times)), 'Pa')
    P_friction_time = Q_(np.zeros(len(times)), 'Pa')
    P_total_time = Q_(np.zeros(len(times)), 'Pa')
    vol_flow_rates_out = Q_(np.zeros(len(times)), 'm**3/s')

    ## Calculate flow rates from scale data + assumed state 
    mdots_DF = Q_(np.zeros(len(times)), 'kg/s')
    for i in range(len(times)):
        mdots_DF[i] = mdot_highpow(times[i])  # Get mass flow rate at this time step
    mdots_products = mass_ratio_products(excess_air) * mdots_DF
    vol_flow_rates_in = mdots_products / density_flue_gas(T_flue_inlet, P_amb)

    ## Initialize Previous Values for Better Convergence 
    T_steel_prev = Q_(np.ones_like(overall_height_arr) * T_amb.magnitude, 'K')
    T_iso_prev = Q_(np.ones_like(overall_height_arr) * T_amb.magnitude, 'K')

    for i in range(len(times)):
        ## CALCULATE EACH SECTIONS CONTRIBUTION TO DRAFT 
        ## Allocate Arrays
        T_flue_av_prof = Q_(np.ones_like(overall_height_arr), 'K')
        T_steel_prof = Q_(np.ones_like(overall_height_arr), 'K')
        T_iso_prof = Q_(np.ones_like(overall_height_arr), 'K')
        T_out_prof = Q_(np.ones_like(overall_height_arr), 'K')
        P_buoyant = Q_(np.zeros_like(overall_height_arr), 'Pa')
        P_friction = Q_(np.zeros_like(overall_height_arr), 'Pa')

        ## BUOYANCY 
        for n in range(len(overall_height_arr)):
            """ Intermediate Values """
            flue_vel = flue_velocity_in(vol_flow_rates_in[i], A_cs)
            htc_inner = inner_heat_transfer_coefficient(d_i_arr[n], flue_vel)
            kA1 = heat_transmission_in(section_height_arr[n], htc_inner, d_i_arr[n])
            C_flue = Cp_flue_gas(T_flue_inlet)
            gamma_1_loop = intermediate_gamma_1(kA1, section_height_arr[0], mdots_products[i], C_flue)

            if n == 0:
                """ CHIMNEY INLET """
                htc_outer = outer_heat_transfer_coefficient(d_o_arr[n], v_wind, T_flue_inlet, T_flue_inlet, T_amb, section_height_arr[n])
                kA2 = heat_transmission_through(section_height_arr[n], d_o_arr[n], d_i_arr[n], conductivity_iso_krueger(T_amb), htc_outer)
                kA3 = heat_transmission_out(section_height_arr[n], htc_outer, d_o_arr[n])
                gamma_2_loop = intermediate_gamma_2(m_steel[n], Cp_steel(), m_iso[n], Cp_iso(), kA2, kA3)

                # Use previous time step values if available
                if i == 0:
                    T_steel_guess = T_amb
                    T_iso_guess = T_amb
                else:
                    T_steel_guess = T_steel_prev[n]
                    T_iso_guess = T_iso_prev[n]

                # ITERATIVE CALCULATION
                T_flue_av_int, T_steel_int, T_iso_int = iterate_flue_gas_temperature(
                    T_flue_inlet, T_steel_guess, T_iso_guess, T_amb, section_height_arr[n], kA1, kA2, kA3, gamma_1_loop, gamma_2_loop
                )

                T_flue_av_prof[n] = T_flue_av_int
                T_steel_prof[n] = T_steel_int
                T_iso_prof[n] = T_iso_int
                T_out_prof[n] = calculate_T_flue_out(gamma_1_loop, section_height_arr[n], T_flue_inlet, T_steel_int)

            else:
                """ CHIMNEY NODE - Receives Flue Gas from Node Below """
                htc_outer = outer_heat_transfer_coefficient(d_o_arr[n], v_wind, T_steel_prof[n-1], T_iso_prof[n-1], T_amb, section_height_arr[n])
                kA2 = heat_transmission_through(section_height_arr[n], d_o_arr[n], d_i_arr[n], conductivity_iso_krueger(T_iso_prof[n-1]), htc_outer)
                kA3 = heat_transmission_out(section_height_arr[n], htc_outer, d_o_arr[n])
                gamma_2_loop = intermediate_gamma_2(m_steel[n], Cp_steel(), m_iso[n], Cp_iso(), kA2, kA3)

                # Use previous time step values if available
                if i == 0:
                    T_steel_guess = T_amb
                    T_iso_guess = T_amb
                else:
                    T_steel_guess = T_steel_prev[n]
                    T_iso_guess = T_iso_prev[n]

                T_flue_av_int, T_steel_int, T_iso_int = iterate_flue_gas_temperature(
                    T_flue_av_prof[n-1], T_steel_guess, T_iso_guess, T_amb, section_height_arr[n], kA1, kA2, kA3, gamma_1_loop, gamma_2_loop
                )

                T_flue_av_prof[n] = T_flue_av_int
                T_steel_prof[n] = T_steel_int
                T_iso_prof[n] = T_iso_int
                T_out_prof[n] = calculate_T_flue_out(gamma_1_loop, section_height_arr[n], T_out_prof[n-1], T_steel_int)

            ## BUOYANCY PRESSURE
            P_buoyant[n] = grav * section_height_arr[n] * (density_amb(T_amb, P_amb) - density_flue_gas(T_flue_av_prof[n], P_amb))

            ## FRICTION PRESSURE
            Re_loop = reynolds(density_flue_gas(T_flue_av_prof[n], P_amb), flue_vel, d_i_arr[n], mu_flue_gas(T_flue_av_prof[n]))
            P_friction[n] = section_pressure_drop(section_height_arr[n], d_i_arr[n], flue_vel, Re_loop, density_flue_gas(T_flue_av_prof[n], P_amb))

        ## WIND PRESSURE
        P_wind = wind_pressure(v_wind, T_amb, P_amb)

        ## TOTAL OUTLET DRAFT
        P_total = -1.0*P_wind + np.sum(P_friction) - np.sum(P_buoyant)

        ## STORE VALUES IN ARRAYS
        T_flue_time[i] = np.copy(T_flue_av_prof)
        T_steel_time[i] = np.copy(T_steel_prof)
        T_iso_time[i] = np.copy(T_iso_prof)
        P_buoyant_time[i] = np.copy(np.sum(P_buoyant))
        P_friction_time[i] = np.copy(np.sum(P_friction))
        P_total_time[i] = P_total.to('Pa')*-1.0

        ## SOLVE FOR VOLUME FLOW RATES TO PLOT
        P_pitot = -1.0*P_wind + P_friction[-1] - P_buoyant[-1]
        vol_flow_rates_out[i] = mdots_products[i] / density_flue_gas(T_flue_av_prof[-1], P_amb+P_pitot)

        ## UPDATE PREVIOUS VALUES FOR NEXT TIME STEP
        T_steel_prev = np.copy(T_steel_prof)
        T_iso_prev = np.copy(T_iso_prof)

        print('Simulation Time:', times[i])
    return times, T_flue_time, vol_flow_rates_out, P_total_time