import os
import sys

# Add the current directory to the Python path to allow for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from display_2_csv import plot_comparison
except ImportError:
    print("Error: Could not import 'plot_comparison' from 'display_2_csv.py'.")
    print("Please ensure 'run_batch_alignment.py' is in the same directory as 'display_2_csv.py' and that the file is named with an underscore, not a hyphen.")
    sys.exit(1)


def run_batch_alignment():
    """
    Runs the alignment and saving process for multiple pairs of simulation and experimental data.

    This script iterates through predefined sets of files, calling the `plot_comparison`
    function from `display_2_csv.py` for each pair. It is configured for batch
    processing to align the start of each simulation file with its corresponding
    experimental data and save the result, without showing interactive plots.
    """
    # --- Configuration ---
    base_dir = "/Users/jacob/Double_pendulum"
    # Directory for raw simulation files (output from Double_Pendulum_Model.py).
    sim_data_dir = os.path.join(base_dir, "Simulation_data")
    # Directory for processed experimental files (output from convert-tracker-data.py).
    exp_data_dir = os.path.join(base_dir, "Data/Experimental_Data_Processed")

    # --- Define File Sets for Alignment ---
    # Each dictionary represents a set of simulations to be aligned against a single experimental file.
    alignment_sets = [
        {
            "sim_files": ["DOP853_172.00s.csv", "GL8_172.00s.csv", "bs_adaptive_free_172.00s.csv"],
            "exp_file": "30-processed.csv"
        },
        {
            "sim_files": ["DOP853_204.00s.csv", "GL8_204.00s.csv", "bs_adaptive_free_204.00s.csv"],
            "exp_file": "60-processed.csv"
        },
        {
            "sim_files": ["DOP853_288.00s.csv", "GL8_288.00s.csv", "bs_adaptive_free_288.00s.csv"],
            "exp_file": "90-processed.csv"
        },
        {
            "sim_files": ["DOP853_292.00s.csv", "GL8_292.00s.csv", "bs_adaptive_free_292.00s.csv"],
            "exp_file": "120-processed.csv"
        },
        {
            "sim_files": ["DOP853_319.00s.csv", "GL8_319.00s.csv", "bs_adaptive_free_319.00s.csv"],
            "exp_file": "150-processed.csv"
        },
    ]

    print("--- Starting Batch Data Alignment ---")

    for i, file_set in enumerate(alignment_sets):
        exp_filepath = os.path.join(exp_data_dir, file_set["exp_file"])
        for sim_file in file_set["sim_files"]:
            print(
                f"\n--- Aligning '{sim_file}' against '{file_set['exp_file']}' ---")
            sim_filepath = os.path.join(sim_data_dir, sim_file)

            # Call the function with non-interactive settings to align and save.
            plot_comparison(
                simulation_filepath=sim_filepath,
                experimental_filepath=exp_filepath,
                align_start=True,
                save_aligned=True,
                show_plot=False  # Explicitly disable plotting for batch mode
            )

    print("\n--- Batch Data Alignment Complete ---")


if __name__ == '__main__':
    run_batch_alignment()
