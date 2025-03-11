import numpy as np

from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity

""" Relative Imports"""
from . import geometry_setup

from .wood_heat_release import mdot_highpow
from .wood_heat_release import mass_ratio_products

from .thermo_properties import density_flue_gas
from .thermo_properties import Cp_flue_gas
from .heat_transfer import conductivity_iso_krueger
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

from .fluid_mechanics import reynolds

from .flow_rates import flue_velocity_in

""" Global Inputs """
# Ambient Temperature/Pressure
T_amb = Q_(15.0, 'degC').to('K')
P_amb = Q_(1.0, 'atm').to('Pa')
# Wind Velocity
v_wind = Q_(0.0, 'm/s')
# Internal Cross-Sectional Area
A_cs = np.pi*(0.5*d_i_arr[0])**2
# gravitational accel
grav = Q_(9.8, 'm/s**2')

""" Simulation time """
t_simulation = 1500.0
times = np.arange(0.0, 1500.0, 1.0)

""" Allocate Time-Dependant Arrays """
T_flue_time = np.zeros(len(times), dtype=object) # Use dtype=object for quantity arrays
T_steel_time = np.zeros(len(times), dtype=object)
T_iso_time = np.zeros(len(times), dtype=object)
P_buoyant_time = np.zeros(len(times), dtype=object)
P_friction_time = np.zeros(len(times), dtype=object)
P_total_time = np.zeros(len(times), dtype=object)
volume_flow_1 = Q_(np.zeros(len(times)), 'm**3/s')
volume_flow_2 = Q_(np.zeros(len(times)), 'm**3/s')
volume_flow_3 = Q_(np.zeros(len(times)), 'm**3/s')
volume_flow_4 = Q_(np.zeros(len(times)), 'm**3/s')
volume_flow_5 = Q_(np.zeros(len(times)), 'm**3/s')
volume_flow_6 = Q_(np.zeros(len(times)), 'm**3/s')
volume_flow_7 = Q_(np.zeros(len(times)), 'm**3/s')

""" Calculate flow rates from scale data + assumed state """
excess_air = 2.25
mdots_DF = Q_(np.zeros(len(times)), 'kg/s')
for i in range(len(times)):
    mdots_DF[i] = mdot_highpow(times[i])  # Get mass flow rate at this time step
mdots_products = mass_ratio_products(excess_air) * mdots_DF
T_flue_inlet = Q_(515.0, 'K')
vol_flow_rates = mdots_products / density_flue_gas(T_flue_inlet)

""" Initialize Previous Values for Better Convergence """
T_steel_prev = Q_(np.ones_like(overall_height_arr) * T_amb.magnitude, 'K')
T_iso_prev = Q_(np.ones_like(overall_height_arr) * T_amb.magnitude, 'K')

for i in range(len(times)):
    """ CALCULATE EACH SECTIONS CONTRIBUTION TO DRAFT """
    ## Allocate Arrays
    T_flue_av_prof = Q_(np.ones_like(overall_height_arr), 'K')
    T_steel_prof = Q_(np.ones_like(overall_height_arr), 'K')
    T_iso_prof = Q_(np.ones_like(overall_height_arr), 'K')
    T_out_prof = Q_(np.ones_like(overall_height_arr), 'K')
    P_buoyant = Q_(np.zeros_like(overall_height_arr), 'Pa')
    P_friction = Q_(np.zeros_like(overall_height_arr), 'Pa')

    """ BUOYANCY """
    for n in range(len(overall_height_arr)):
        """ Intermediate Values """
        flue_vel = flue_velocity_in(vol_flow_rates[i], A_cs)
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

        """ BUOYANCY PRESSURE """
        P_buoyant[n] = grav * section_height_arr[n] * (density_amb(T_amb) - density_flue_gas(T_flue_av_prof[n]))

        """ FRICTION PRESSURE """
        Re_loop = reynolds(density_flue_gas(T_flue_av_prof[n]), flue_vel, d_i_arr[n], mu_flue_gas(T_flue_av_prof[n]))
        P_friction[n] = section_pressure_drop(section_height_arr[n], d_i_arr[n], flue_vel, Re_loop, density_flue_gas(T_flue_av_prof[n]))

    """ WIND PRESSURE """
    P_wind = wind_pressure(v_wind, T_amb, P_amb)

    """ TOTAL OUTLET DRAFT """
    P_total = -1.0*P_wind + np.sum(P_friction) - np.sum(P_buoyant)

    """ STORE VALUES IN ARRAYS """
    T_flue_time[i] = np.copy(T_flue_av_prof)
    T_steel_time[i] = np.copy(T_steel_prof)
    T_iso_time[i] = np.copy(T_iso_prof)
    P_buoyant_time[i] = np.copy(P_buoyant)
    P_friction_time[i] = np.copy(P_friction)
    P_total_time[i] = P_total


    """ SOLVE FOR VOLUME FLOW RATES TO PLOT """
    # Static Draft/Friction Pressure at Pitot tube
    P_pitot = -1.0 * P_wind + P_friction[-1] - P_buoyant[-1]
    # Velocities with different assumptions
    pitot_vel_1 = flue_velocity_general(-1.0*P_total, -1.0*P_pitot, density_flue_gas(T_flue_inlet), flue_velocity_in(vol_flow_rates[i], A_cs), density_flue_gas(T_flue_av_prof[-2]), grav, overall_height_arr[-2])
    pitot_vel_2 = flue_velocity_ignore_gravity(-1.0*P_total, -1.0*P_pitot, density_flue_gas(T_flue_inlet), flue_velocity_in(vol_flow_rates[i], A_cs), density_flue_gas(T_flue_av_prof[-2]))
    pitot_vel_3 = flue_velocity_ignore_draft(-1.0*P_total, density_flue_gas(T_flue_inlet), flue_velocity_in(vol_flow_rates[i], A_cs), density_flue_gas(T_flue_av_prof[-2]), grav, overall_height_arr[-2])
    pitot_vel_4 = flue_velocity_ignore_both(-1.0*P_total, density_flue_gas(T_flue_inlet), flue_velocity_in(vol_flow_rates[i], A_cs), density_flue_gas(T_flue_av_prof[-2]))
    
    pitot_vel_5 = flue_velocity_assume_static(-1.0*P_total, -1.0*P_pitot, density_flue_gas(T_flue_av_prof[-2]), grav, overall_height_arr[-2])
    pitot_vel_6 = flue_velocity_static_nograv(-1.0*P_total, -1.0*P_pitot, density_flue_gas(T_flue_av_prof[-2]))
    pitot_vel_7 = flue_velocity_static_nograv(-1.0*P_total, Q_(0.0,'Pa'), density_flue_gas(T_flue_av_prof[-2]))
    
    # Volume Flow Rates
    volume_flow_1[i] = pitot_vel_1*A_cs
    volume_flow_2[i] = pitot_vel_2*A_cs
    volume_flow_3[i] = pitot_vel_3*A_cs
    volume_flow_4[i] = pitot_vel_4*A_cs
    volume_flow_5[i] = pitot_vel_5*A_cs
    volume_flow_6[i] = pitot_vel_6*A_cs
    volume_flow_7[i] = pitot_vel_7*A_cs
    
    
    """ UPDATE PREVIOUS VALUES FOR NEXT TIME STEP """
    T_steel_prev = np.copy(T_steel_prof)
    T_iso_prev = np.copy(T_iso_prof)

    print('Time:', times[i])