import numpy as np

from .units import ureg, Q_

""" Relative Imports"""
from .heat_transfer import section_stovepipe_mass
from .heat_transfer import section_insulation_mass
from .thermo_properties import rho_iso

""" Overall Height and Section Heights """
overall_height_arr = Q_(np.linspace(0.0, 13.0, 14), 'ft')
section_height_arr = (overall_height_arr[2]-overall_height_arr[1])*np.ones_like(overall_height_arr)

""" Section Diameters:
First 6 feet, only 6 inch ID steel
Last 7 feet, packed insulation (2 inches)
"""
d_i_arr = Q_(2.0, 'in')*np.ones_like(overall_height_arr)
A_cs = np.pi*(0.5*d_i_arr[0])**2
steel_thickness = Q_(0.0239,'in') # Guage 24 Steel
steel_density = Q_(7832.0, 'kg/m**3')
d_o_arr = d_i_arr + steel_thickness
## Last 7 feet

insulation_thickness = Q_(4.0, 'in') # Assumed

d_o_arr[7:] = d_o_arr[7:]+insulation_thickness

""" Section Masses.
First 6 feet - steel only
Last 7 feet - Insulation
"""
m_steel = Q_(np.zeros_like(overall_height_arr), 'kg')
m_iso = Q_(np.zeros_like(overall_height_arr), 'kg')
for i in range(len(overall_height_arr)):
    if i < 7:
        ## Uninsulated
        m_steel[i] = section_stovepipe_mass(d_i_arr[i], section_height_arr[i], steel_thickness, steel_density)
        m_iso[i] = Q_(0.0, 'kg')
    else:
        ## Insulated
        m_steel[i] = section_stovepipe_mass(d_i_arr[i], section_height_arr[i], steel_thickness, steel_density)
        m_iso[i] = section_insulation_mass(d_i_arr[i], d_o_arr[i], section_height_arr[i], rho_iso())