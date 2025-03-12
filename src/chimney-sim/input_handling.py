import argparse
import yaml

from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity

## IMPORT MAIN CALCULATION FUNCTION
from .run_simulation import run_simulation

def parse_arguments():
    """
    Parses command-line arguments for the simulation.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments, including the YAML input file path.
    """
    parser = argparse.ArgumentParser(description="Run the chimney draft simulation.")
    parser.add_argument('--input', type=str, required=True, help="Path to the YAML input file.")
    return parser.parse_args()

def read_yaml_file(file_path):
    """
    Reads simulation inputs from a YAML file.

    Parameters
    ----------
    file_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        A dictionary containing structured simulation input values.
    """
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)  # Load YAML data into a dictionary
    return data

def main():
    """
    Main function to execute multiple simulations for different T_flue_inlet and excess_air values.

    Reads input parameters from a YAML file, extracts relevant input values,
    and runs batch simulations for each combination of T_flue_inlet and excess_air.
    """
    args = parse_arguments()  # Parse command-line arguments
    config = read_yaml_file(args.config)  # Read YAML configuration file

    # Extract single-value inputs from the config file
    t_simulation = config['simulation']['t_simulation']
    timestep = config['simulation']['timestep']
    T_amb = Q_(config['environment']['T_amb'], 'K')
    P_amb = Q_(config['environment']['P_amb'], 'Pa')
    v_wind = Q_(config['environment']['v_wind'], 'm/s')

    # Get lists of T_flue_inlet and excess_air values from the config
    T_flue_inlet_list = config['combustion']['T_flue_inlet']
    excess_air_list = config['combustion']['excess_air']

    # Run simulations for each combination of T_flue_inlet and excess_air
    for T_flue_inlet in T_flue_inlet_list:
        for excess_air in excess_air_list:
            run_simulation(t_simulation, timestep, T_amb, P_amb, v_wind, Q_(T_flue_inlet, 'K'), excess_air)

if __name__ == "__main__":
    main()