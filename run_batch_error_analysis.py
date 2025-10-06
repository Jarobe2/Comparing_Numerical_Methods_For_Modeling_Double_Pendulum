import os
import sys

# Add the current directory to the Python path to allow for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from calculate_error import calculate_and_plot_error, merge_computation_times
except ImportError:
    print("Error: Could not import 'calculate_and_plot_error'.")
    print("Please ensure 'run_batch_error_analysis.py' is in the same directory as 'calculate_error.py'.")
    sys.exit(1)


def run_batch_analysis():
    """
    Runs error analysis for multiple pairs of simulation and experimental data.

    This script iterates through a predefined list of file pairs and calls the
    `calculate_and_plot_error` function with settings suitable for batch processing
    (e.g., no plots, logging enabled).
    """
    # --- Configuration ---
    # Define the base directories for your data.
    base_dir = "/Users/jacob/Double_pendulum/Data"
    # Directory for aligned simulation files (output from display-2-csv.py with alignment).
    sim_data_dir = os.path.join(
        base_dir, "Simulation_Data_Processed")
    # Directory for processed experimental files (output from convert-tracker-data.py).
    exp_data_dir = os.path.join(base_dir, "Experimental_Data_Processed")

    # --- Define File Pairs for Analysis ---
    # List of tuples, where each tuple is (simulation_filename, experimental_filename).
    # These files are expected to be in the directories defined above.
    analysis_pairs = [
        # --- 30 Degrees ---
        ("DOP853_172.00s-aligned.csv", "30-processed.csv"),
        ("GL8_172.00s-aligned.csv", "30-processed.csv"),
        ("bs_adaptive_free_172.00s-aligned.csv", "30-processed.csv"),
        # --- 60 Degrees ---
        ("DOP853_204.00s-aligned.csv", "60-processed.csv"),
        ("GL8_204.00s-aligned.csv", "60-processed.csv"),
        ("bs_adaptive_free_204.00s-aligned.csv", "60-processed.csv"),
        # --- 90 Degrees ---
        ("DOP853_288.00s-aligned.csv", "90-processed.csv"),
        ("GL8_288.00s-aligned.csv", "90-processed.csv"),
        ("bs_adaptive_free_288.00s-aligned.csv", "90-processed.csv"),
        # --- 120 Degrees ---
        ("DOP853_292.00s-aligned.csv", "120-processed.csv"),
        ("GL8_292.00s-aligned.csv", "120-processed.csv"),
        ("bs_adaptive_free_292.00s-aligned.csv", "120-processed.csv"),
        # --- 150 Degrees ---
        ("DOP853_319.00s-aligned.csv", "150-processed.csv"),
        ("GL8_319.00s-aligned.csv", "150-processed.csv"),
        ("bs_adaptive_free_319.00s-aligned.csv", "150-processed.csv"),
    ]

    print("--- Starting Batch Error Analysis ---")

    for i, (sim_file, exp_file) in enumerate(analysis_pairs):
        print(f"\n--- Processing Pair {i+1}/{len(analysis_pairs)} ---")
        print(f"  Simulation:   {sim_file}")
        print(f"  Experimental: {exp_file}")
        print("------------------------------------")

        sim_filepath = os.path.join(sim_data_dir, sim_file)
        exp_filepath = os.path.join(exp_data_dir, exp_file)

        # Check if files exist before running the analysis
        if not os.path.exists(sim_filepath):
            print(
                f"Error: Simulation file not found at '{sim_filepath}'. Skipping pair.")
            continue
        if not os.path.exists(exp_filepath):
            print(
                f"Error: Experimental file not found at '{exp_filepath}'. Skipping pair.")
            continue

        # Call the analysis function with non-interactive settings.
        calculate_and_plot_error(
            simulation_filepath=sim_filepath,
            experimental_filepath=exp_filepath,
            log_results_to_csv=True,       # Log results to the summary CSV
            plot_error=False,              # Disable plotting for batch mode
            analyze_stabilization=True,    # Run divergence analysis
            validate_interpolation=True   # Optional: set to True if needed
        )

    print("\n--- Batch Error Analysis Loop Complete ---")

    # --- Final Step: Merge Computation Times ---
    # After all errors have been calculated and logged, merge the timing data.
    rmse_file = os.path.join(
        base_dir, "Error_Analysis_Data/RMSE_data.csv")
    times_file = os.path.join(
        "/Users/jacob/Double_pendulum", "Simulation_data/computation_times.csv")
    merge_computation_times(rmse_file, times_file)
    print("\n--- Full Batch Error Analysis Complete ---")


if __name__ == '__main__':
    run_batch_analysis()
