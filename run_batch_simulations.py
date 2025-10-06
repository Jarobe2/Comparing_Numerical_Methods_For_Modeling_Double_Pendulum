import os
import sys

# Add the current directory to the Python path to allow for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # We import the plotting function which also handles the simulation and saving
    from Double_Pendulum_Model import plot_double_pendulum_path
except ImportError:
    print("Error: Could not import 'plot_double_pendulum_path'.")
    print("Please ensure 'run_batch_simulations.py' is in the same directory as 'Double_Pendulum_Model.py'.")
    sys.exit(1)


def run_batch_simulations():
    """
    Runs multiple double pendulum simulations with different initial conditions and solvers.

    This script iterates through a predefined list of configurations and calls the
    `plot_double_pendulum_path` function with settings suitable for batch processing
    (e.g., saving data, disabling plots).
    """
    # --- Configuration ---
    # Define the numerical methods (solvers) to be used for the simulations.
    methods_to_run = ['DOP853', 'GL8', 'bs_adaptive_free']

    # Define the initial conditions. Each dictionary represents a unique scenario.
    # 'angle_rad' is the initial angle for bob 1. Bob 2 is always at 0 rad.
    # 'sim_time' is the total duration for that specific simulation.
    simulation_configs = [
        {'angle_rad': 0.523598775598, 'sim_time': 172},  # 30 degrees
        {'angle_rad': 1.04719755,     'sim_time': 204},  # 60 degrees
        {'angle_rad': 1.57079633,     'sim_time': 288},  # 90 degrees
        {'angle_rad': 2.0943951,      'sim_time': 292},  # 120 degrees
        {'angle_rad': 2.61799388,     'sim_time': 319},  # 150 degrees
    ]

    print("--- Starting Batch Simulation Generation ---")
    total_sims = len(methods_to_run) * len(simulation_configs)
    sim_count = 0

    # Loop through each configuration and each solver
    for config in simulation_configs:
        for method in methods_to_run:
            sim_count += 1
            print(f"\n--- Running Simulation {sim_count}/{total_sims} ---")
            print(f"  Method:       {method}")
            print(f"  Initial Angle: {config['angle_rad']:.4f} rad")
            print(f"  Sim Time:     {config['sim_time']}s")
            print("------------------------------------")

            # Call the simulation function with non-interactive settings
            plot_double_pendulum_path(
                theta1_init=config['angle_rad'],
                theta2_init=0.0,  # Bob 2 starts at 0 rad as requested
                method_name=method,
                sim_time=config['sim_time'],
                save_data=True,  # Ensure the data is saved to a CSV
                show_plot=False  # Disable the plot window for batch mode
            )

    print("\n--- Batch Simulation Generation Complete ---")


if __name__ == '__main__':
    run_batch_simulations()
