import argparse
import yaml
import os
import csv

from .units import ureg, Q_

# Import main calculation function
from .run_simulation import run_simulation

def parse_arguments():
    """
    Parses command-line arguments for the simulation.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments, including the YAML input file path and output directory.
    """
    parser = argparse.ArgumentParser(description="Run the chimney draft simulation.")
    parser.add_argument('--input', type=str, required=True, help="Path to the YAML input file.")
    parser.add_argument('--output', type=str, required=True, help="Path to the output results directory.")
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

def save_temperature_profiles_to_csv(output_dir, T_flue_inlet, excess_air, times, T_flue):
    """
    Saves the temperature profiles over time to a CSV file.

    Parameters
    ----------
    output_dir : str
        Directory where the CSV file will be saved.
    T_flue_inlet : float
        Flue gas inlet temperature (K).
    excess_air : float
        Excess air ratio.
    times : np.ndarray
        Time steps.
    T_flue : np.ndarray
        Flue gas temperature profile.

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    filename = f"NO_INSUlATION_TemperatureProfile_{T_flue_inlet:.0f}K_{excess_air:.1f}ExcessAir.csv"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (s)", "T_flue (K)"])  # Header
        for i in range(len(times)):
            writer.writerow([times[i], T_flue[i].magnitude])

    print(f"Saved temperature profile to {filepath}")

def save_scalar_values_to_csv(output_dir, T_flue_inlet, excess_air, times, vol_flow, P_total):
    """
    Saves scalar values (volume flow rate, total pressure) over time to a CSV file.

    Parameters
    ----------
    output_dir : str
        Directory where the CSV file will be saved.
    T_flue_inlet : float
        Flue gas inlet temperature (K).
    excess_air : float
        Excess air ratio.
    times : np.ndarray
        Time steps.
    vol_flow : np.ndarray
        Volume flow rates at the outlet.
    P_total : np.ndarray
        Total pressure over time.

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    filename = f"NO_INSUlATION_ScalarValues_{T_flue_inlet:.0f}K_{excess_air:.1f}ExcessAir.csv"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (s)", "Vol_Flow_Out (m³/s)", "P_total (Pa)"])  # Header
        for i in range(len(times)):
            writer.writerow([times[i], vol_flow[i].to('m**3/s').magnitude, P_total[i].to('Pa').magnitude])

    print(f"Saved scalar values to {filepath}")

def main():
    """
    Main function to execute multiple simulations for different T_flue_inlet and excess_air values.

    Reads input parameters from a YAML file, extracts relevant input values,
    runs batch simulations for each combination of T_flue_inlet and excess_air,
    and saves results to CSV files.
    """
    args = parse_arguments()  # Parse command-line arguments
    input_data = read_yaml_file(args.input)  # Read YAML input file
    output_dir = args.output  # Get output directory

    # Extract single-value inputs from the config file
    t_simulation = input_data['simulation']['t_simulation']
    timestep = input_data['simulation']['timestep']
    T_amb = Q_(input_data['environment']['T_amb'], 'degC').to('K')
    P_amb = Q_(input_data['environment']['P_amb'], 'Pa')
    v_wind = Q_(input_data['environment']['v_wind'], 'm/s')

    # Get lists of T_flue_inlet and excess_air values from the config
    T_flue_inlet_list = input_data['combustion']['T_flue_inlet']
    excess_air_list = input_data['combustion']['excess_air']

    # Run simulations for each combination of T_flue_inlet and excess_air
    for T_flue_inlet in T_flue_inlet_list:
        for excess_air in excess_air_list:
            # Run simulation
            times, T_flue, vol_flow, P_total = run_simulation(
                t_simulation, timestep, T_amb, P_amb, v_wind, Q_(T_flue_inlet, 'K'), excess_air
            )

            # Save temperature profiles to CSV
            save_temperature_profiles_to_csv(output_dir, T_flue_inlet, excess_air, times, T_flue)

            # Save scalar values to CSV
            save_scalar_values_to_csv(output_dir, T_flue_inlet, excess_air, times, vol_flow, P_total)

if __name__ == "__main__":
    main()
