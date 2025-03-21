import pytest
import pint
from chimney_sim.thermo_properties import density_flue_gas

# Set up Pint unit registry
ureg = pint.UnitRegistry()

def test_density_flue_gas():
    """Test density_flue_gas with known values."""
    T = 500 * ureg.kelvin  # Mid-range temperature
    P = 101325 * ureg.pascal  # Atmospheric pressure

    # Expected density using the ideal gas law: ρ = (P * M) / (R * T)
    M = 0.029559  # kg/mol (molar mass of flue gas)
    R = 8.314  # J/(mol*K)
    expected_rho = (P.magnitude * M) / (R * T.magnitude)  # Density in kg/m³

    result = density_flue_gas(T, P)
    
    # Assert the computed density is close to the expected value
    assert result.to('kg/m**3').magnitude == pytest.approx(expected_rho, rel=1e-5)

def test_density_units():
    """Ensure the function correctly handles Pint quantities."""
    T = 600 * ureg.kelvin
    P = 150000 * ureg.pascal  # 1.5 atm

    result = density_flue_gas(T, P)

    # Check that the returned value is a Pint Quantity with correct units
    assert isinstance(result, pint.Quantity)
    assert result.check('[mass] / [length] ** 3')  # kg/m³

def test_density_extreme_values():
    """Test the function with the expected range of temperatures and pressures."""
    T_low = 300 * ureg.kelvin  # Lower bound of expected range
    T_high = 700 * ureg.kelvin  # Upper bound of expected range
    P_low = 50000 * ureg.pascal  # 0.5 atm (reasonable low pressure)
    P_high = 300000 * ureg.pascal  # 3 atm (reasonable high pressure)

    assert density_flue_gas(T_low, P_high).magnitude > 0
    assert density_flue_gas(T_high, P_low).magnitude > 0

def test_density_invalid_inputs():
    """Ensure the function raises errors for invalid inputs."""
    with pytest.raises(AttributeError):
        density_flue_gas(500, 101325)  # Missing Pint units

    with pytest.raises(ValueError):
        density_flue_gas(250 * ureg.kelvin, 101325 * ureg.pascal)  # Below expected range

    with pytest.raises(ValueError):
        density_flue_gas(800 * ureg.kelvin, 101325 * ureg.pascal)  # Above expected range

if __name__ == "__main__":
    pytest.main()
