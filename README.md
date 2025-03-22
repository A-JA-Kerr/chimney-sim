# Chimney Simulation Package

This projet is a Python package which can be used to simulate natural draft in the chimney of a residential wood stove.

## Getting Started
The package is broken down into the following modules:
- flow_rates.py
- fluid_mechanics.py
- geometry_setup.py
- heat_transfer.py
- run_simulation.py
- thermo_properties.py
- units.py
- wood_heat_release.py

The package can be interacted with using the command line. Simulation inputs are managed using a yaml file, which is included in /config/.

An example of running the package is given here:
python -m chimney_sim --input inputs_path\simulation_inputs.yaml --output outputs_path\

### Prerequisites
What software or dependencies are required?
- Python 3.x
- Git
- Any required libraries (e.g., `numpy`, `scipy`, `cantera`)

### Installation
This project can be installed from this Github repository into your local environment by using:

pip install git+https://github.com/A-JA-Kerr/chimney-sim.git