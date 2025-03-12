import argparse
import yaml

## IMPORT MAIN CALCULATION FUNCTION
from .run_simulation import run_simulation

""" Simulation Inputs """
# Ambient Temperature/Pressure
T_amb = Q_(15.0, 'degC').to('K')
P_amb = Q_(1.0, 'atm').to('Pa')
# Wind Velocity
v_wind = Q_(0.0, 'm/s')
# Simulation time
t_simulation = 3000.0
timestep = 1.0
# Combustion inputs
T_flue_inlet = Q_(515.0, 'K')
excess_air = 1.5

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

    Reads input parameters from a YAML file and runs batch simulations.
    """
    args = parse_arguments()  # Parse command-line arguments
    config = read_yaml_file(args.config)  # Read YAML configuration file

    # Get lists of T_flue_inlet and excess_air values from the config
    T_flue_inlet_list = config['combustion']['T_flue_inlet']
    excess_air_list = config['combustion']['excess_air']

    # Run simulations for each combination of T_flue_inlet and excess_air
    for T_flue_inlet in T_flue_inlet_list:
        for excess_air in excess_air_list:
            run_simulation(T_flue_inlet, excess_air, config)

if __name__ == "__main__":
    main()